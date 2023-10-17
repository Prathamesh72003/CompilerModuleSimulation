from flask import Flask, render_template, request, jsonify
from typing import List
import re
import os
import pandas as pd
import json

app = Flask(__name__)

# Tokenizing
class Tokenizer:
    def __init__(self, filepath) -> None:
        self.programlines = self.readInput(filepath)
        
    def readInput(self, filepath: str) -> List[str]:
        '''
        To read the input source program from a text file
        '''
        with open(filepath, 'r') as f:
            programlines = f.readlines()
            programlines = [line.replace("\n", "").strip() for line in programlines]
            programlines = [line.split() for line in programlines]
            return programlines
    
    def start_scanning(self, outputFile) -> None:
        '''
        Scans and write the output to a file
        '''
        with open(outputFile, 'w') as f:
            for lines in self.programlines:
                for word in lines:
                    token = self.scan(word)
                    line = f'< {token[1]}, {token[0]} > \n' 
                    self.writeOutput(filepath=outputFile, line=line)
    
    def scan(self, lex: str) -> List[str]:
        '''
        Actual scanning lexical analysis happens here         
        '''
        keywords: List[str] = ["main", "int", "float", "double", "long",
                               "short", "string", "char", "if", "else", "while",
                               "do", "break", "continue"]
        operators: List[str] = ["+", "-", "*", "/", "<", ">", "=", "|", "&", "^", "~", "%"]
        punctuations: List[str] = ["{", "}", "(", ")", ";", "[", "]", ".", "&"]
        identifiers = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
        constants = r'\b[0-9][0-9]*\b'
        
        if lex in keywords:
            return [lex, 'Keyword'] 
        elif lex in operators:
            return [lex, 'Operator'] 
        elif lex in punctuations:
            return [lex, 'Punctuation']
        elif re.findall(identifiers, lex):
            return [lex, 'Identifier']
        elif re.findall(constants, lex):
            return [lex, 'Constant']
    
    def writeOutput(self, filepath: str, line: str) -> None:
        with open(filepath, 'a') as f:
            f.write(line)

#SYMBOLTABLE
class SymbolTable:
    def __init__(self, inputPath) -> None:
        self.data = {
            'id': [],
            'datatype': [],
            'input_value': [],
            'return_type': [],
            'arguments': []
        }
        self.dataKeys = list(self.data.keys())
        self.dataTypes = ['string', 'char', 'int', 'float', 'short', 'long', 'double']
        self.currType = ''
        self.symTab = ['null'] * 5
        self.sourceCode = self.readInput(inputPath)

    def readInput(self, filepath) -> List[str]:
        '''
        To read the input source program from a text file
        '''
        with open(filepath, 'r') as f:
            programLines = f.readlines()
            programLines = [line.replace('\n', '').strip() for line in programLines]
            programLines = [line.split() for line in programLines]
        return programLines

    def generate(self) -> dict:
        '''
        Actual Symbol Table generation
        '''
        for line in self.sourceCode:
            self.symTab = ['null'] * 5
            for idx, token in enumerate(line):
                if token in self.dataTypes:
                    try:
                        if not (line[idx - 1] == '(' or line[idx - 1] == ','):
                            self.currType = self.symTab[1] = token
                            self.symTab[0] = line[idx + 1]
                    except IndexError:
                        pass
                elif token == ',':
                    if not line[idx + 1] in self.dataTypes:
                        for index in range(len(self.symTab)):
                            self.data[self.dataKeys[index]].append(self.symTab[index])
                        self.symTab[0] = line[idx + 1]
                elif token == '=':
                    value = line[idx + 1]
                    if self.currType == 'char' or self.currType == 'string':
                        value = value[1:-1]
                    self.symTab[2] = value
                elif token == '(':
                    self.symTab[3] = self.currType
                    arguments = line[idx + 1: line.index(')')]
                    arguments = ''.join(arguments)
                    self.symTab[4] = arguments
                elif token == ')':
                    self.symTab[1] = 'null'
            for index in range(len(self.symTab)):
                self.data[self.dataKeys[index]].append(self.symTab[index])
        return self.data

    def getSymbolTable(self) -> pd.DataFrame:
        '''
        To get the symbol table as a DataFrame
        '''
        data = self.generate()
        df = pd.DataFrame(data)
        return df

    def writeOutput(self, filepath: str) -> pd.DataFrame:
        '''
        To write the DataFrame into a txt file
        '''
        dataframe = self.getSymbolTable()
        with open(filepath, 'w') as f:
            f.write(dataframe.to_string())
        return dataframe

# ThreeCodeGen
class IntermediateCodeGenerator:
    def __init__(self, expression):
        self.stack = []
        self.operators = ['+', '-', '*', '/']
        self.statement = re.split('([+\-*/])', expression)
        self.temp_count = 1

    def code_generator(self):
        while len(self.statement) > 1:
            self.compare()
            if len(self.statement) == 1:
                return '\n'.join(f't{i} = {stmt}' for i, stmt in enumerate(self.stack, start=1))
        return 'Unable to generate three-address code.'

    def compare(self):
        for word in self.statement:
            if word in self.operators:
                self.stack.append(''.join(self.statement[:3]))
                self.statement.pop(0)
                self.statement.pop(0)
                self.statement[0] = 't' + str(self.temp_count)
                self.temp_count += 1

@app.route('/')
def index():
    return render_template('index.html')

# route for tokenSection
@app.route('/tokenize', methods=['POST'])
def tokenize():
    file = request.files['file']
    
    # Ensure the 'uploads' and 'output' directories exist
    current_directory = os.path.dirname(os.path.abspath(__file__))
    upload_directory = os.path.join(current_directory, 'uploads')
    output_directory = os.path.join(current_directory, 'output')
    
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    file_path = os.path.join(upload_directory, file.filename)
    file.save(file_path)

    scanner = Tokenizer(filepath=file_path)
    output_file_path = os.path.join(output_directory, 'tokenization_output.txt')
    scanner.start_scanning(output_file_path)

    # Read the tokenization output and store it in a list
    with open(output_file_path, 'r') as f:
        tokenization_output = f.readlines()

    # Prepare the data as a list of dictionaries
    tokenization_data = []
    for line in tokenization_output:
        # Assuming tokenization_output.txt file has one token per line
        tokenization_data.append({'token': line.strip()})

    # Convert the data to JSON and return
    # print(tokenization_data)
    return json.dumps(tokenization_data)

# route for symbolTable
@app.route('/generate_symbol_table', methods=['POST'])
def generate_symbol_table():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)

        # Generate the symbol table using the provided file path
        generator = SymbolTable(file_path)
        symbol_table_data = generator.generate()

        # Delete the uploaded file after processing
        os.remove(file_path)

        return jsonify(symbol_table_data)

# route for three code
@app.route('/generate_code', methods=['POST'])
def generate_code():
    expression = request.form['expression']
    generator = IntermediateCodeGenerator(expression)
    result = generator.code_generator()

    if result:
        return result
    else:
        return 'Unable to generate three-address code.'

def cal_follow(s, productions, first):
    follow = set()
    if len(s)!=1 :
        return {}
    if(s == list(productions.keys())[0]):
        follow.add('$') 
    
    for i in productions:
        for j in range(len(productions[i])):
            if(s in productions[i][j]):
                idx = productions[i][j].index(s)
                
                if(idx == len(productions[i][j])-1):
                    if(productions[i][j][idx] == i):
                        break
                    else:
                        f = cal_follow(i, productions, first)
                        for x in f:
                            follow.add(x)
                else:
                    while(idx != len(productions[i][j]) - 1):
                        idx += 1
                        if(not productions[i][j][idx].isupper()):
                            follow.add(productions[i][j][idx])
                            break
                        else:
                            f = cal_first(productions[i][j][idx], productions)
                            
                            if('ε' not in f):
                                for x in f:
                                    follow.add(x)
                                break
                            elif('ε' in f and idx != len(productions[i][j])-1):
                                f.remove('ε')
                                for k in f:
                                    follow.add(k)
                            
                            elif('ε' in f and idx == len(productions[i][j])-1):
                                f.remove('ε')
                                for k in f:
                                    follow.add(k)
                                
                                f = cal_follow(i, productions, first)
                                for x in f:
                                    follow.add(x)
                            
    return follow
   
def cal_first(s, productions):
    
    first = set()
    
    for i in range(len(productions[s])):
        
        for j in range(len(productions[s][i])):
            
            c = productions[s][i][j]
            
            if(c.isupper()):
                f = cal_first(c, productions)
                if('ε' not in f):
                    for k in f:
                        first.add(k)
                    break
                else:
                    if(j == len(productions[s][i])-1):
                        for k in f:
                            first.add(k)
                    else:
                        f.remove('ε')
                        for k in f:
                            first.add(k)
            else:
                first.add(c)
                break
                
    return first


@app.route('/startproj', methods=['GET', 'POST'])
def display_first_follow():
    first = {}
    follow = {}

    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            # Save the uploaded file to the server
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)

            # Parse the grammar from the uploaded file
            productions = {}
            with open(file_path, 'r') as grammar_file:
                for prod in grammar_file:
                    l = re.split("( /->/\n/)*", prod)
                    m = []
                    for i in l:
                        if (i == "" or i == None or i == '\n' or i == " " or i == "-" or i == ">"):
                            pass
                        else:
                            m.append(i)
                    
                    left_prod = m.pop(0)
                    right_prod = []
                    t = []
                    
                    for j in m:
                        if(j != '|'):
                            t.append(j)
                        else:
                            right_prod.append(t)
                            t = []
                    
                    right_prod.append(t)
                    productions[left_prod] = right_prod

            # Calculate FIRST and FOLLOW sets
            for s in productions.keys():
                first[s] = cal_first(s, productions)

            for s in productions.keys():
                follow[s] = cal_follow(s, productions, first)

            # Remove the uploaded file
            os.remove(file_path)

    return render_template('index.html', first=first, follow=follow)


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

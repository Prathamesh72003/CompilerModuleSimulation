from flask import Flask, render_template, request, jsonify
from typing import List
import re
import os
import pandas as pd

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
        operators: List[str] = ["+", "-", "*", "/", "<", ">", "=", "|", "&"]
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

    # Read the tokenization output and send it back to the client
    with open(output_file_path, 'r') as f:
        tokenization_output = f.read()

    return tokenization_output

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


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

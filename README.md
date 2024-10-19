# Compiler Module Simulation

This project simulates various phases of a compiler, including tokenization, syntax analysis, and more. The application is built using Flask and provides an interactive web interface to visualize the compilation process.

## Getting Started

Follow these steps to set up the project locally and run the application.

### Prerequisites

- Python 3.x installed on your machine
- `pip` (Python package manager)

### Setup Instructions

1. **Clone the repository**:

   Open your terminal and run the following command to clone the project repository:

   ```bash
   git clone https://github.com/Prathamesh72003/CompilerModuleSimulation.git
   ``` 

2. **Navigate to the project directory**:

   ```bash
   cd CompilerModuleSimulation
   ```

3. **Create a virtual environment:**:

   To isolate project dependencies, create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

4. **Install dependencies:**:

   Use pip to install the necessary packages from the requirements.txt file:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Flask application:**:

   Start the Flask development server by running:

   ```bash
   python app.py
   ```

   The application should now be running on http://127.0.0.1:5000.

### Using the Application

1. Open your browser and visit: [http://127.0.0.1:5000/startproj](http://127.0.0.1:5000/startproj)

2. In the navigation bar, click the first option: **Tokenization**. This will format the frontend for you.

3. Once the formatting is done, explore the various tabs in the navbar, each representing different phases of compilation.

4. To see a simulation in action, upload the sample files provided in the `upload` directory. The results for each phase will be displayed in the respective tabs.

Enjoy exploring the different phases of the compiler simulation!

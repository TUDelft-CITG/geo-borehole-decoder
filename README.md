## Tool to digitize borehole logs 

Borehole descriptions are detailed descriptions of the layer structure in the subsurface, and this data is often not available in a structured form. The goal of this project is to develop a tool that should ensure automatic data extraction. The tool uses a combination of artificial intelligence, natural language processing and optical character recognition technologies. This will make it possible to automatically convert the information from the PDF files into shareable, structured data. 

## Requirements to use the scripts

Follow the below given steps to make your environment ready to use the package 'bhdecode'.

### Steps

- Install Python (version >=3.7 & < 3.11)

- Install the latest Microsoft Visual C++ Redistributable from [here](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170)

- Poppler is a PDF rendering library used to manipulate PDF files. To install poppler binaries in Windows follow the below steps:

    - Download and unzip 'poppler-22.04.0' [from](https://github.com/oschwartz10612/poppler-windows/releases/download/v22.04.0-0/Release-22.04.0-0.zip) into your 'C:\Program Files\', thus the path is 'C:\Program Files\poppler-22.04.0'
    - Add 'C:\Program Files\poppler-22.04.0\Library\bin' to your system PATH by doing the following: Search for 'Edit the system environment variables' in Windows Search, click on 'Environment Variables'.  Within 'System variables', look for PATH and double-click on it, click on New, then add 'C:\Program Files\poppler-22.04.0\Library\bin', click OK.

- Tesseract-OCR is used for extracting text from pdf files. To install Tesseract-OCR in Windows follow these steps:

    - Install tesseract-OCR using windows installer available [here](https://github.com/UB-Mannheim/tesseract/wiki) 
    - While installing make sure that the Destination is set to'C:\Program Files\Tesseract-OCR'.

- It is recommended to create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) and activate it.

- Use 'pip install -r requirements.txt' to install the required python packages (upgrade pip if necessary: 'python -m pip install --upgrade pip').


## How to use the scripts

The example script (main.py) along with two pdf files (/input_pdf) are saved in the samples folder. Follow the below instructions to process the example pdfs and get the ouput excel files.

- Go to the samples directory ('cd samples')

- Save pdf files into 'input_pdf' folder (there are two example pdf files in the folder).

- Run 'main.py' ('python .\main.py'). 
    - The trained classification model is saved as 'model.h5'. This will be used in the sample script ('main.py') for checking whether the page is a borehole log.

- Two new folders will be created. 
    - 'pdf_images': pdf pages are converted into jpeg images and saved here.
    - 'Output': the output excel files are saved here.

- Check the results in the 'output' folder
    - For each of the pdf file, if the page is identified as a borehole log by the classification model, an excel file with two sheets are created. In the first sheet of the excel file, metadata about the borehole such as borehole name, geographical coordinates, city, country etc are saved. In the second sheet, the extracted depth and description are saved.
    The flow of the code is as given below: 
    
    ![Flow of the code](geo-borehole-decoder/docs/media/codeflow.jpg)

## How to (re)-train the classification model

- Save the input files in two folders: [bhole, non_bhole]. 
- Follow the code train_model-> train_classification.py to train the model on the training set.
- Evaluate the results on the test set (train_classification.py). 
- If the results are satisfactory, save the model and replace 'model.h5' in samples directory with the new model.

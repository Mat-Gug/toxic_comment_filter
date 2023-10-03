# Deep Learning Project: Toxic Comment Filter :iphone: :mag:

Hello and welcome! Thank you for taking the time to explore this project :blush:

The present project aims to develop a Deep Learning multi-label classification model for filtering toxic comments within social network platforms.

To achieve this goal, the project is divided into several distinct steps:

* **Data Exploration**: In this phase, we will delve into the dataset to gain a better understanding of the data;
* **Data Pre-processing**: The collected dataset will undergo a comprehensive data cleaning and pre-processing pipeline. This includes tasks such as text tokenization to transform the raw data into a format suitable for training Deep Learning models;
* **Model Building and Evaluation**: The next step is to train different Deep Learning models on the preprocessed text data, capable of multi-label classification.

You can access the project by referring to the `Toxic Comments Filter Project.ipynb` notebook.<br>
Additionally, `utils.py` houses custom functions utilized throughout the project, and `requirements.txt` can be employed to install the necessary dependencies in the project's virtual environment. Detailed instructions for setting up the virtual environment are provided below.

## Setting Up a Virtual Environment and Installing Dependencies

Before running the project, it's considered a best practice to create a virtual environment and install the required dependencies. This helps isolate project-specific dependencies from system-wide Python packages. To achieve this, follow these steps:

1. **Create a Virtual Environment:**
- For Windows:
  - To create a virtual environment with the default Python version:
    ```
    python -m venv toxic_venv
    ```
  - To create a virtual environment with a specific Python version, such as Python 3.11, replace `python` with `py` followed by the desired Python version:
    ```
    py -3.11 -m venv toxic_venv
    ```
- For macOS and Linux:
  ```
  python3 -m venv toxic_venv
  ```
2. **Activate the Virtual Environment:**
- For Windows (Command Prompt):
  ```
  toxic_venv\Scripts\activate
  ```
- For Windows (Git Bash):
  ```
  source toxic_venv/Scripts/activate
  ```
- For macOS and Linux:
  ```
  source toxic_venv/bin/activate
  ```
3. **Clone the Repository:**
```
git clone https://github.com/Mat-Gug/toxic_comment_filter.git
```
4. **Navigate to the Project Directory:**
```
cd toxic_comment_filter
```
5. **Install Required Dependencies:**
```
pip install -r requirements.txt
```
6. **Create an IPython Kernel for Jupyter Notebook:**

After activating your virtual environment, run the following command to create an IPython kernel for Jupyter Notebook:
```
python -m ipykernel install --user --name=toxic_venv_kernel
```
If you don't have `ipykernel` installed, you can do it by running the following command:
```
pip install ipykernel
```
7. **Deactivate the Virtual Environment:**

Whenever you're done working on the project, you can deactivate the virtual environment:
```
deactivate
```
By following these steps, you'll have your project set up in an isolated virtual environment with all the required dependencies installed, and you'll be able to use Jupyter Notebook with your project-specific kernel. This is very helpful to ensure that the project runs consistently and without conflicts.

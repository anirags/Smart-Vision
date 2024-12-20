from flask import Flask,session, render_template, request, redirect, url_for, flash, send_file, make_response
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from driver import main
import io
from pathlib import Path

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER,exist_ok=True)
REQUIRED_COLUMNS = {'Title','TestSteps','ID'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_columns(df):
    return REQUIRED_COLUMNS.issubset(df.columns)


def generate_pie_chart(data):
    
    data = data[~(data['Category'] == "Invalid Data")]
    categories = data['Category']
    tc_count = data['Test_Cases_Count']
    colors = data['colors']
    explode = data['explode']
    textprops = {"fontsize":15}
    plt.figure(figsize=(8,6))
    
    plt.pie(
        tc_count,
        labels=categories,
        explode=explode,
        colors = colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops=textprops,
        wedgeprops = dict(edgecolor='white')
        )
    
    img=BytesIO()
    plt.savefig(img, format='png')    
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8') 
    


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            if file.filename.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            elif file.filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            
            try:
                base=os.path.basename(file.filename)
                filename = os.path.splitext(base)[0]
                file_name = Path(filename).stem
                session['input_filename'] = file_name
                 
                final_output_df,duplicate_data_summary,almost_duplicate_summary,highlysimilar_summary, moderate_duplicate_summary,almost_unique_summary,unique_summary,invalid_summary = main(df,filename)
                #final_output_df= main(df,filename)
               
                
                print("--------------------------------------------------")
                
                pie_chart = generate_pie_chart(final_output_df)
               
                total_test_case_count = final_output_df['Test_Cases_Count'].sum()
                
                return render_template('ui.html',
                                       final_output_df=final_output_df,
                                       duplicate_data_summary = duplicate_data_summary,
                                       almost_duplicate_summary = almost_duplicate_summary,
                                       highlysimilar_summary = highlysimilar_summary,
                                       moderate_duplicate_summary = moderate_duplicate_summary,
                                       almost_unique_summary = almost_unique_summary,
                                       unique_summary = unique_summary,
                                       invalid_summary = invalid_summary, 
                                       pie_chart=pie_chart,
                                       total_test_case_count = total_test_case_count,
                                       execution_done=True  )
              
            except Exception as e:
                return redirect(request.url)
                
              
        else:
            flash('Allowed file types are .xlsx, .csv')
            return redirect(request.url)
    return render_template('upload_ui.html')

@app.route('/details')
def show_details():
    return "Detailed summary would be displayed here."


@app.route('/download')
def download_all_files():
    # Path to the Excel file stored on your local disk
    dire = os.getcwd()
    filename = session.get('input_filename')
    path_to_file = os.path.join(dire, "output",filename+".xlsx")
   
    # Send the file to the client
    return send_file(path_to_file, as_attachment=True, download_name="output_"+filename+".xlsx")

    
@app.route('/download_file/<category>',methods=['GET'])
def excel_file_download(category):
    file_directory = { 
        "Duplicate":"DuplicateTC.xlsx",
        "Almost_Duplicate":"Almost_DuplicatesTC.xlsx",
        "Highly_Similar": "Highly_SimilarTC.xlsx",
        "Moderate_Duplicate":"Moderate_DuplicateTC.xlsx",
        "Almost_Unique":"Almost_UniqueTC.xlsx",
        "Unique": "UniqueTC.xlsx",
        "Invalid_Data":"InvalidTC.xlsx"
        
        }
    category_name = file_directory.get(category)
    dire = os.getcwd()
    filename = session.get("input_filename")
    full_filename = filename +"_"+category_name
    output = 'output'+"_"+filename
    path_to_file = os.path.join(dire,output,full_filename)

    if path_to_file and os.path.exists(path_to_file):
        return send_file(path_to_file,as_attachment=True,download_name="output_"+full_filename)
    else:
        return "File Not Found.-",404
    

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, use_reloader=False)

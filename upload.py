from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
def upload_dataset2(app):
    print("Request received!")

    if 'file' not in request.files:
        print("Error: No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        print("Error: No selected file")
        return jsonify({"error": "No selected file"}), 400

    # Save file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    print(f"File saved to {file_path}")

    # Read dataset
    try:
        sep=","
        df = pd.read_csv(file_path)
        print(df.shape)
        if df.shape[1] ==1:
            sep = "\t"
            df = pd.read_csv(file_path,sep=sep)

            if df.shape[1] ==1:
                sep=" "
                df = pd.read_csv(file_path,sep=sep)

        try: 
            float(df.columns[0])
            df = pd.read_csv(file_path,sep=sep,header= None)
        except: 
            pass
            
        # print(df)
        


        print("Dataset loaded successfully!")
    except Exception as e:
        print("Error reading CSV:", str(e))
        return jsonify({"error": "Failed to read CSV file"}), 400

    # Return the first 5 rows
    return df
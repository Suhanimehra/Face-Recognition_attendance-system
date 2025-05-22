# Face Recognition Based Attendance System

This project implements a face recognition-based attendance system using Python and OpenCV. It provides a web interface to mark and display attendance records using facial recognition.

## 📁 Project Structure
├──app.py # Main Flask application
├── train.py # Script to train the face recognizer
├── trainer.yml # Trained face recognizer model
├── labels.pickle # Encoded labels for known faces
├── requirements.txt # Python dependencies
├── templates/
│ └── attendance.html # Webpage to show attendance
├── static/
│ └── style.css # Styling for the attendance page
└── README.md # Project documentation


## 🚀 Features

- Face recognition using OpenCV
- Attendance recording via webcam
- Web interface to view attendance
- Easy-to-train system with new faces

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-attendance-system.git
   cd face-attendance-system

2. Create a virtual environment (optional but recommended):
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install dependencies:
   pip install -r requirements.txt

   
##  🧠 Training the Model
1. Add training images in a folder structure like:
    dataset/
    ├── person1/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── person2/
        ├── img1.jpg
        └── img2.jpg

2. Run the training script:
   python train.py

This will generate trainer.yml and labels.pickle.

##  💻 Running the App
Start the Flask app:

python app.py


##  📄 License
This project is open-source and available under the MIT License.


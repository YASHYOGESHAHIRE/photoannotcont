Object Detection API
Overview
This project is a FastAPI-based Object Detection API that processes images to detect specified objects using the Google Gemini 1.5 Flash model. It supports multiple annotation output formats (YOLO, Pascal VOC, COCO) and visualization styles (overlay, border). Users can upload images, specify target objects, and receive processed images with annotations in their preferred format, delivered as a downloadable ZIP file. The API is designed for scalability, includes CORS support, and is optimized for deployment on platforms like Render.
The application is ideal for computer vision tasks, such as generating annotated datasets for machine learning models. It includes a web interface for user interaction and a background thread to keep the server warm, ensuring quick response times in production environments.
Features

Image Processing: Upload multiple images for object detection using Google Gemini's vision capabilities.
Annotation Formats: Supports YOLO, Pascal VOC, and COCO formats for object detection annotations.
Visualization Styles: Offers overlay and border styles for visualizing detected objects.
Asynchronous Processing: Built with FastAPI for efficient handling of concurrent image uploads.
CORS Support: Allows cross-origin requests for seamless integration with front-end applications.
Web Interface: Includes static HTML pages for homepage, documentation, preview, and more.
Temporary File Management: Automatically cleans up temporary uploads to optimize storage.
Server Warmth: Background thread pings the health endpoint to prevent cold starts in production.
Environment Configuration: Uses .env for secure API key management.

Technologies Used

Backend: FastAPI, Python 3.8+
Computer Vision: OpenCV, Google Gemini 1.5 Flash
File Handling: aio reposefiles, pathlib, shutil
Environment Management: python-dotenv
Web Serving: Uvicorn
Static Files: HTML, served via FastAPI's StaticFiles
Dependencies: requests, tqdm, xml.etree.ElementTree, numpy

Installation
Prerequisites

Python 3.8+
Google Gemini API Key (Obtain from Google Cloud or relevant provider)
Git for cloning the repository
pip for installing dependencies

Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/object-detection-api.git
cd object-detection-api


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the project root and add your Google Gemini API key:
GOOGLE_API_KEY=your_google_api_key_here


Run the Application:
python main.py

The server will start on http://localhost:8000.


Dependencies
Ensure you have a requirements.txt file with the following:
fastapi
uvicorn
python-dotenv
google-generativeai
opencv-python
aiofiles
requests
tqdm
numpy

Install them using:
pip install fastapi uvicorn python-dotenv google-generativeai opencv-python aiofiles requests tqdm numpy

Usage
API Endpoints

POST /upload-images/: Upload images, specify target objects, output format, and visualisation style. Returns a ZIP file containing processed images and annotations.

Parameters:
files: List of image files (jpg, jpeg, png, bmp)
target_object: String specifying the object to detect (e.g., "car")
output_format: Annotation format ("yolo", "pascal_voc", "coco")
visualization_style: Visualisation style ("overlay", "border")


Response: JSON with a hex-encoded ZIP file containing results.


GET /formats/: Returns available annotation formats and visualisation styles.

GET /health: Health check endpoint, returns {"status": "healthy"}.

GET /ping: Returns {"status": "pong"}.

GET /: Serves the homepage (homepage.html).

GET /preview: Serves the preview page (index.html).

GET /documentation: Serves the documentation page (docpage.html).

GET /github: Serves the GitHub page (git.html).

GET /developer: Serves the about page (aboutpage.html).

GET /counterpreview: Serves the counter preview page (counterpage.html).


Example Request
To upload images and process them for car detection in YOLO format with overlay visualisation:
curl -X POST "http://localhost:8000/upload-images/" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "target_object=car" \
  -F "output_format=yolo" \
  -F "visualization_style=overlay"

Web Interface
Access the homepage at http://localhost:8000/ to interact with the application via a browser. The static pages provide a user-friendly interface for uploading images and viewing results.
Project Structure
object-detection-api/
│
├── main.py                  # Main FastAPI application
├── static/                  # Static HTML files for web interface
│   ├── homepage.html
│   ├── index.html
│   ├── docpage.html
│   ├── git.html
│   ├── aboutpage.html
│   ├── counterpage.html
├── temp_uploads/            # Temporary directory for uploaded images
├── .env                     # Environment variables (not tracked in Git)
├── requirements.txt         # Python dependencies
├── README.md                # This file

Deployment
The application is designed for deployment on platforms like Render or Heroku. To deploy:

Set Up Environment Variables: Add GOOGLE_API_KEY in your platform's environment settings from Google AI Studio.
Configure Port: Ensure the app runs on the port specified by the platform (e.g., $PORT on Render).
Static Files: Ensure the static/ directory is included in the deployment.
Keep-Alive: The background thread (keep_server_warm) ensures the server remains responsive by pinging the /health endpoint every 5 minutes.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, please contact yashyogeshahire@gmail.com or open an issue on GitHub.

Built with ❤️ by Yash Yogesh Ahire

# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
import aiofiles
from typing import List
import tempfile
import cv2
import numpy as np
import google.generativeai as genai
import json
import xml.etree.ElementTree as ET
import base64
from datetime import datetime
import re
from tqdm import tqdm
from fastapi.staticfiles import StaticFiles
import threading
import time
import requests
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Google Gemini
GOOGLE_API_KEY = "AIzaSyDwyTeQ3mY5WeSy8iF78fVd7qC0nYomPBw"  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# Create temporary directories
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



from datetime import datetime, timedelta

# Add this function before your routes, around line 200-250
def cleanup_temp_uploads(base_dir, exclude_folder=None):
    """Clean up temp uploads, excluding the currently created folder."""
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            
            # Skip the currently created folder
            if exclude_folder and os.path.samefile(item_path, exclude_folder):
                continue
            
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    except Exception as e:
        print(f"Error during temp folder cleanup: {e}")



def upload_to_gemini(image_path, mime_type="image/jpeg"):
    """Upload image to Gemini."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return {"mime_type": mime_type, "data": image_data}
    except Exception as e:
        print(f"Error uploading to Gemini: {e}")
        return None

def extract_object_info(response_text):
    """Extract object information from Gemini response."""
    try:
        # Find JSON content using regex
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        print(f"Error extracting object info: {e}")
        return None

class AnnotationExporter:
    @staticmethod
    def to_yolo(object_info, image_shape, class_mapping):
        """Convert to YOLO format."""
        height, width = image_shape[:2]
        yolo_annotations = []
        
        for obj in object_info.get('objects', []):
            box = obj.get('boundingBox', {})
            class_id = class_mapping.get(obj.get('name', '').lower(), 0)
            
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            x_center = box.get('left', 0) + (box.get('width', 0) / 2)
            y_center = box.get('top', 0) + (box.get('height', 0) / 2)
            
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {box.get('width', 0)} {box.get('height', 0)}")
        
        return "\n".join(yolo_annotations)

    @staticmethod
    def to_pascal_voc(object_info, image_path, image_shape):
        """Convert to Pascal VOC format."""
        height, width = image_shape[:2]
        
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = str(Path(image_path).parent.name)
        ET.SubElement(root, "filename").text = str(Path(image_path).name)
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(image_shape[2])
        
        for obj in object_info.get('objects', []):
            obj_elem = ET.SubElement(root, "object")
            ET.SubElement(obj_elem, "name").text = obj.get('name', '')
            
            box = obj.get('boundingBox', {})
            bbox = ET.SubElement(obj_elem, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(int(box.get('left', 0) * width))
            ET.SubElement(bbox, "ymin").text = str(int(box.get('top', 0) * height))
            ET.SubElement(bbox, "xmax").text = str(int((box.get('left', 0) + box.get('width', 0)) * width))
            ET.SubElement(bbox, "ymax").text = str(int((box.get('top', 0) + box.get('height', 0)) * height))
        
        return ET.tostring(root, encoding='unicode')

    @staticmethod
    def to_coco(images_info, class_mapping):
        """Convert to COCO format."""
        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        for name, class_id in class_mapping.items():
            coco_dict["categories"].append({
                "id": class_id,
                "name": name,
                "supercategory": "object"
            })
        
        annotation_id = 1
        for image_id, (image_path, object_info) in enumerate(images_info.items(), 1):
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            coco_dict["images"].append({
                "id": image_id,
                "file_name": str(Path(image_path).name),
                "width": width,
                "height": height
            })
            
            for obj in object_info.get('objects', []):
                box = obj.get('boundingBox', {})
                x = box.get('left', 0) * width
                y = box.get('top', 0) * height
                w = box.get('width', 0) * width
                h = box.get('height', 0) * height
                
                coco_dict["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_mapping.get(obj.get('name', '').lower(), 0),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1
        
        return coco_dict

async def process_single_image(image_path, target_object, visualization_style='overlay'):
    """Process a single image using Gemini."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None

        file = upload_to_gemini(image_path)
        if not file:
            return None, None

        prompt = [
            f"""Please analyze this image and provide detailed object detection information for the target objects: {target_object} in the following JSON format:
            {{
                "objects": [
                    {{
                        "name": "object_name",
                        "confidence": 0.95,
                        "boundingBox": {{
                            "left": 0.1,
                            "top": 0.2,
                            "width": 0.3,
                            "height": 0.4
                        }},
                      
                    }}
                ],
              
            }}""",
            file,
        ]

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        response = model.generate_content(prompt)
        object_info = extract_object_info(response.text)
        print(object_info)
        if object_info and object_info.get('objects'):
            result_image = draw_object_visualization(image, object_info, visualization_style)
            return result_image, object_info
        
        return image, None

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def draw_object_visualization(image, object_info, visualization_style='overlay', confidence_threshold=0.5):
    """Draw object detection visualization."""
    output = image.copy()
    height, width = image.shape[:2]

    if visualization_style == 'overlay':
        overlay = np.zeros_like(image, dtype=np.uint8)
        for obj in object_info.get('objects', []):
            confidence = obj.get('confidence', 0)
            if confidence < confidence_threshold:
                continue

            box = obj.get('boundingBox', {})
            x1 = int(box.get('left', 0) * width)
            y1 = int(box.get('top', 0) * height)
            x2 = int((box.get('left', 0) + box.get('width', 0)) * width)
            y2 = int((box.get('top', 0) + box.get('height', 0)) * height)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            label = f"{obj.get('name', 'Unknown')} ({confidence:.2f})"
            cv2.putText(output, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        alpha = 0.3
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    elif visualization_style == 'border':
        for obj in object_info.get('objects', []):
            confidence = obj.get('confidence', 0)
            if confidence < confidence_threshold:
                continue

            box = obj.get('boundingBox', {})
            x1 = int(box.get('left', 0) * width)
            y1 = int(box.get('top', 0) * height)
            x2 = int((box.get('left', 0) + box.get('width', 0)) * width)
            y2 = int((box.get('top', 0) + box.get('height', 0)) * height)

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{obj.get('name', 'Unknown')} ({confidence:.2f})"
            cv2.putText(output, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output

async def process_folder_with_format(folder_path, target_object, output_format='yolo', visualization_style='overlay'):
    """Process folder and save annotations in specified format."""
    folder_path = Path(folder_path)
    output_base = folder_path.parent / f"{folder_path.name}_{output_format}"
    
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir()

    # Create format-specific folders
    if output_format in ['yolo', 'pascal_voc']:
        (output_base / 'images').mkdir()
        (output_base / ('labels' if output_format == 'yolo' else 'annotations')).mkdir()

    # Process images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in folder_path.glob('*') if f.suffix.lower() in image_extensions]
    
    class_mapping = {}
    class_id = 0
    images_info = {}

    for img_path in image_files:
        result_image, object_info = await process_single_image(
            str(img_path), 
            target_object, 
            visualization_style
        )
        
        if result_image is None or object_info is None:
            continue

        # Update class mapping
        for obj in object_info.get('objects', []):
            obj_name = obj.get('name', '').lower()
            if obj_name not in class_mapping:
                class_mapping[obj_name] = class_id
                class_id += 1

        # Save processed image and annotations
        output_image_path = output_base / 'images' / img_path.name
        cv2.imwrite(str(output_image_path), result_image)

        if output_format == 'yolo':
            yolo_txt = AnnotationExporter.to_yolo(object_info, result_image.shape, class_mapping)
            txt_path = output_base / 'labels' / f"{img_path.stem}.txt"
            with open(txt_path, 'w') as f:
                f.write(yolo_txt)

        elif output_format == 'pascal_voc':
            voc_xml = AnnotationExporter.to_pascal_voc(object_info, str(img_path), result_image.shape)
            xml_path = output_base / 'annotations' / f"{img_path.stem}.xml"
            with open(xml_path, 'w') as f:
                f.write(voc_xml)

        elif output_format == 'coco':
            images_info[str(img_path)] = object_info

    # Save format-specific files
    if output_format == 'yolo':
        with open(output_base / 'classes.txt', 'w') as f:
            for name, _ in sorted(class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{name}\n")

    elif output_format == 'coco':
        coco_json = AnnotationExporter.to_coco(images_info, class_mapping)
        with open(output_base / 'annotations.json', 'w') as f:
            json.dump(coco_json, f, indent=2)

    return output_base

@app.post("/upload-images/")
async def upload_images(
    files: List[UploadFile] = File(...),
    target_object: str = Form(...),
    output_format: str = Form(...),
    visualization_style: str = Form(...)
):
    try:
        # Create temporary folder
        cleanup_temp_uploads(UPLOAD_DIR)
        temp_folder = Path(tempfile.mkdtemp(dir=UPLOAD_DIR))
        
        # Save uploaded files
        for file in files:
            file_path = temp_folder / file.filename
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
        
        # Process the folder
        output_folder = await process_folder_with_format(
            str(temp_folder),
            target_object,
            output_format,
            visualization_style
        )
        
        # Create zip file
        output_zip = shutil.make_archive(
            str(temp_folder / "results"),
            'zip',
            output_folder
        )
        
        # Read zip file
        with open(output_zip, 'rb') as f:
            zip_contents = f.read()
            
        # Clean up
        shutil.rmtree(temp_folder)
        
        return JSONResponse(
            content={
                "message": "Processing complete",
                "zip_contents": zip_contents.hex()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/formats/")
async def get_formats():
    """Get available formats and visualization styles."""
    return {
        "formats": [
            {"id": "yolo", "name": "YOLO"},
          
        ],
        "visualization_styles": [
            {"id": "overlay", "name": "Overlay"},
            {"id": "border", "name": "Border"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/ping")
def ping():
    return {"status": "pong"}

# Background thread to keep the server warm

def keep_server_warm():
    while True:
        try:
            # Use the internal network address for Render
            requests.get("http://localhost:8000/health", timeout=10)
        except Exception as e:
            print(f"Self-ping failed: {e}")
        time.sleep(300)  # 5 minutes

threading.Thread(target=keep_server_warm, daemon=True).start()

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
@app.get("/")
async def serve_homepage():
    """Serve the homepage HTML file."""
    return FileResponse("static/homepage.html")
@app.get("/preview")
async def serve_index():
    """Serve the index HTML file."""
    return FileResponse('static/index.html')

@app.get("/documentation")
async def serve_index():
    """Serve the index HTML file."""
    return FileResponse('static/docpage.html')
@app.get("/github")
async def serve_index():
    """Serve the index HTML file."""
    return FileResponse('static/git.html')




@app.get("/developer")
async def serve_index():
    """Serve the index HTML file."""
    return FileResponse('static/aboutpage.html')

@app.get("/counterpreview")
async def serve_index():
    """Serve the index HTML file."""
    return FileResponse('static/counterpage.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
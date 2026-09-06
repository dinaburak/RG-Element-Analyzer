# Rhythmic Gymnastics Element Analyzer

A Python-based computer vision application that analyzes rhythmic gymnastics
elements from video by tracking body landmarks and joint movement. The system
detects held positions, calculates joint angles, and identifies performed
elements using reference data stored in PostgreSQL.

## Features

- Processes rhythmic gymnastics videos frame by frame using OpenCV
- Detects body landmarks using MediaPipe Pose Landmarker
- Visualizes tracked joints and body segments directly on the video
- Detects when an athlete enters and holds a position
- Excludes preparation and exit movement from pose measurements
- Calculates average joint angles during the detected hold
- Compares measured angles with reference values stored in PostgreSQL
- Uses configurable angle tolerances to account for variation in body position
- Identifies the corresponding rhythmic gymnastics element
- Returns the identified element's base value
- Uses FastAPI and SQLModel for backend and database integration

# Demo

Demo video coming soon.

## Technologies

- Python
- MediaPipe
- OpenCV
- FastAPI
- PostgreSQL
- SQLModel
- NumPy
- pandas

## Project Structure

    main.py
    FastAPI backend and database model

    pose_test.py
    Video processing, landmark tracking, joint-angle calculation,
    hold detection, and visualization

    element_identifier.py
    Compares measured joint angles with reference data stored in
    PostgreSQL to identify the performed element

    pose_landmarker_full.task
    MediaPipe pose landmark model

    requirements.txt
    Python dependencies required to run the project

## Installation

Clone the repository and install the required Python packages:

    pip install -r requirements.txt

## Running the Project

Start the FastAPI backend:

    uvicorn main:app --reload

Run the video analysis:

    python pose_test.py

## Current Limitations

- Element identification currently uses knee and hip angles
- Reference measurements are currently available for a limited number of elements
- Measurements use 2D pose landmarks and may be affected by camera angle
- Current hold detection is primarily designed for held balance elements

## Future Development

- Expand identification to elements of all categories (turns and jumps as well)
- Integrate video upload and analysis directly into the FastAPI backend
- Add real-time camera analysis to identify and evaluate rhythmic gymnastics elements as they are performed

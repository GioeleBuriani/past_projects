# Automatic generation of a Sierpiński triangle by a simple geometric rule

<br>
<img src="./Sierpiński%20-%20Animation.gif" width="300">
<br>

**Period**: Apr 2023

## Context
I saw a video on Instagram of a guy showing how, by progressively repeating a simple geometric operation drawing points inside of a triangle it was possible to generate a Sierpiński triangle. Therefore I decided to automate it with a Python script to see the result.

## Project Description
This project automates the creation of the Sierpiński triangle through a Python script, inspired by a social media demonstration. The essence of the script is to iteratively perform a simple geometric operation—plotting points within a triangle—to progressively reveal the complex pattern of the Sierpiński triangle.

The script operates by initiating with a single point inside a triangle and then continuously relocating this point halfway towards one of the triangle’s vertices, chosen randomly. With each iteration, the point's new position contributes to forming the overall fractal pattern. This process is repeated multiple times, which allows the distinct triangular voids characteristic of the Sierpiński triangle to emerge clearly.

The Python implementation focuses on simplicity and visual clarity, making it accessible for educational purposes, illustrating how simple iterative steps can lead to the creation of complex mathematical and natural patterns.

## Files
- **Sierpiński - Animation.gif**: Animation of the final result
- **Sierpiński - Code.py**: Python code for the project

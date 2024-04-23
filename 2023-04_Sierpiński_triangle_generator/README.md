# Automatic generation of a Sierpiński triangle by a simple geometric rule

<br>
<img src="./Sierpiński%20-%20Animation.gif" width="300">
<br>

**Period**: Apr 2023

## Context
I saw a video on Instagram of a guy showing how, by progressively repeating a simple geometric operation drawing points inside of a triangle it was possible to generate a Sierpiński triangle. Therefore I decided to automate it with a Python script to see the result.

## Project Description
This project involves a Python script designed to automate the creation of the Sierpiński triangle, a complex fractal pattern derived from a simple geometric operation. The inspiration for this automation came from a visual demonstration seen on social media.

The method starts by selecting an initial point randomly within the confines of a triangle. From there, the algorithm enters a loop where it repeatedly selects one of the triangle’s vertices at random. The next point is then determined by finding the midpoint between the previously placed point and the chosen vertex. This process iterates, with each new point plotted being the midpoint between the last point and a randomly selected vertex. Over many iterations, this method results in the formation of the Sierpiński triangle, with its characteristic pattern of recursively nested triangular voids becoming apparent.

The script is designed with simplicity in mind, focusing on the geometric operations needed to visually render the Sierpiński triangle through repeated iterations. The project demonstrates a methodical approach to generating complex fractal patterns from basic geometric principles.

## Files
- **Sierpiński - Animation.gif**: Animation of the final result
- **Sierpiński - Code.py**: Python code for the project

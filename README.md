# DanChoeScripting

### What is this programming Language?

I implemented **procedural programming language**, which will be called DanChoeScript, by using Python and PLY (parser generator).
This complier reads input text file, then it creates blocks depends on type of blocks and statical scope by reading curly brackets. 
Each block has an array of children nodes. If block is function type, it only runs when it is called by callee.

For error handling, there are only two types of error, a semantic error and syntax error.
A semantic error occurs when the line does not contain a syntax error, but one of the "must" conditions given above is violated when evaluating it. If the line contains a semantic error, it prints out SEMANTIC ERROR. Otherwise, it prints SYNTAX ERROR.

### How to run?

It requires to install PLY and Python 3.0+

> python3 main.py inputfile.txt


### Screenshots

![screenshot1](https://github.com/dan-choe/DanChoeScripting/blob/master/screenshot.PNG "DanChoeScripting")

## License
Copyright [2018] [Dan Choe](https://github.com/dan-choe)

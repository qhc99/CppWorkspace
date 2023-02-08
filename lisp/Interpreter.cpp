//
// Created by QC on 2023-01-29.
//
#include "Interpreter.h"
#include "InputPort.h"
#include <iostream>
#include <memory.h>
#include <memory>
#include <string>

using std::string;
shared_ptr<Value> Interpreter::parse(shared_ptr<Value> source,
                                     Interpreter &interpreter) {
  // TODO unfinished
  return std::make_shared<Value>(nullptr);
}

void Interpreter::repl() {
  const string promp{"lisp++>"};
  InputPort in_port{std::cin};
  std::cout << ">>lisp++ interpreter<<" << std::endl;
  while (true) {
    
  }
  /*
  String prompt = "Jis.py>";
  InputPort inPort = new InputPort(System.in);
  System.out.println("Jispy version 2.0");
  while(true) {
    try{
      System.out.print(prompt);
      var x = parse(inPort);
      if(x == null){ continue; }
      else if(x.equals(eof)){ continue; }
      evalAndPrint(x);
      flushConsole();
    }
    catch(Exception e){
      consoleWriteLine(e.toString());
      flushConsole();
    }
  }
  */
}
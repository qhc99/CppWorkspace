cmake_minimum_required(VERSION 3.20)

project(inet_programing C)

set(CMAKE_C_STANDARD 17)


add_executable(simple_client simplec.c)
add_executable(simple_server simples.c)

add_executable(tcpserver tcpserver.c etcp.h library/tcp_server.c utils.c)
add_executable(tcpclient tcpclient.c etcp.h library/tcp_client.c utils.c)


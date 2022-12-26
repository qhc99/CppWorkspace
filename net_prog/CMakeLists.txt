cmake_minimum_required(VERSION 3.20)

project(net_programing C)

set(CMAKE_C_STANDARD 17)


add_executable(simple_client simplec.c)
add_executable(simple_server simples.c)

add_executable(tcp_server tcp_server_template.c etcp.h library/tcp_server.c utils.c)
add_executable(tcp_client tcp_client_template.c etcp.h library/tcp_client.c utils.c)

add_executable(udp_server etcp.h utils.c udp_server_template.c)
add_executable(udp_client etcp.h utils.c udp_client_template.c)


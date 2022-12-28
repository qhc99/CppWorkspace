cmake_minimum_required(VERSION 3.20)

project(net_programing C)

set(CMAKE_C_STANDARD 17)


add_executable(simple_client simplec.c)
add_executable(simple_server simples.c)

add_executable(tcp_server tcp_server_template.c etcp.h lib/tcp_server.c utils.c lib/readn.c lib/readvrec.c)
add_executable(tcp_client tcp_client_template.c etcp.h lib/tcp_client.c utils.c lib/readn.c)

add_executable(udp_server etcp.h utils.c udp_server_template.c lib/udp_server.c lib/readn.c lib/readvrec.c)
add_executable(udp_client etcp.h utils.c udp_client_template.c lib/udp_client.c lib/readn.c)

add_executable(vrs etcp.h utils.c lib/tcp_server.c lib/readn.c lib/readvrec.c vrs.c)
add_executable(vrc etcp.h utils.c lib/tcp_client.c lib/readn.c lib/readvrec.c vrc.c)


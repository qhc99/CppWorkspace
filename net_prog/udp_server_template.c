//
// Created by Nathan on 2022-12-25.
//
#include "etcp.h"

/**
 * stub
 * @param s
 * @param peerp
 */
static void server(SOCKET s, struct sockaddr_in *peerp) {
    struct sockaddr_in peer;
    int peerlen;
    char buf[ 1 ];
    for ( ;; )
    {
        peerlen = sizeof( peer );
        if ( recvfrom( s, buf, sizeof( buf ), 0,
                       ( struct sockaddr * )&peer, &peerlen ) < 0 )
            error( 1, errno, "recvfrom failed" );
        if ( sendto( s, "hello, world\n", 13, 0,
                     ( struct sockaddr * )&peer, peerlen ) < 0 )
            error( 1, errno, "sendto failed" );
    }
}

char *program_name;

int main(int argc, char **argv) {
    struct sockaddr_in local;
    char *hname;
    char *sname;
    SOCKET s;
    INIT();
    if (argc == 2) {
        hname = NULL;
        sname = argv[1];
    } else {
        hname = argv[1];
        sname = argv[2];
    }
    set_address(hname, sname, &local, "udp");
    s = socket(AF_INET, SOCK_DGRAM, 0);
    if (!isvalidsock(s))
        error(1, errno, "socket call failed");
    if (bind(s, (struct sockaddr *) &local,
             sizeof(local)))
        error(1, errno, "bind failed");
    server(s, &local);
    EXIT(0);
}
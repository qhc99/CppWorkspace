//
// Created by Nathan on 2022-12-26.
//
#include "etcp.h"

/**
 * stub
 * @param s
 * @param peerp
 */
static void client(SOCKET s, struct sockaddr_in *peerp) {
    ssize_t rc;
    socklen_t peerlen;
    char buf[ 120 ];
    peerlen = sizeof( *peerp );
    if ( sendto( s, "", 1, 0, ( struct sockaddr * )peerp,
                 peerlen ) < 0 )
        error( 1, errno, "sendto failed" );
    rc = recvfrom( s, buf, sizeof( buf ), 0,
                   ( struct sockaddr * )peerp, &peerlen );
    if ( rc >= 0 )
        write( 1, buf, rc );
    else
        error( 1, errno, "recvfrom failed" );
}

char *program_name;

int main(int argc, char **argv) {
    struct sockaddr_in peer;
    SOCKET s;
    INIT();
    set_address(argv[1], argv[2], &peer, "udp");
    s = socket(AF_INET, SOCK_DGRAM, 0);
    if (!isvalidsock(s))
        error(1, errno, "socket call failed");
    client(s, &peer);
    exit(0);
}
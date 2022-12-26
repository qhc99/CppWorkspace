#include "etcp.h"

/**
 * stub
 * @param s
 * @param peerp
 */
static void server(SOCKET s, struct sockaddr_in *peerp) {
    send(s, "hello, world\n", 13, 0);
}

char *program_name;


int main(int argc, char **argv) {

    struct sockaddr_in local;
    struct sockaddr_in peer;
    char *hname;
    char *sname;
    socklen_t peerlen;
    SOCKET s1;
    SOCKET s;
    const int on = 1;

    INIT();

    if (argc == 2) {
        hname = NULL;
        sname = argv[1];
    } else {
        hname = argv[1];
        sname = argv[2];
    }

    set_address(hname, sname, &local, "tcp");

    s = socket(AF_INET, SOCK_STREAM, 0);

    if (!isvalidsock(s))
        error(1, errno, "socket call failed");

    if (setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on,
                   sizeof(on)))
        error(1, errno, "setsockopt failed");

    if (bind(s, (struct sockaddr *) &local,
             sizeof(local)))
        error(1, errno, "bind failed");

    if (listen(s, NLISTEN))
        error(1, errno, "listen failed");

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
    do {

        peerlen = sizeof(peer);
        s1 = accept(s, (struct sockaddr *) &peer, &peerlen);
        if (!isvalidsock(s1))
            error(1, errno, "accept failed");
        server(s1, &peer);
        CLOSE(s1);
    } while (1);
#pragma clang diagnostic pop
    EXIT(0);
}
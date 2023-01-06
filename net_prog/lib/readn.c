//
// Created by Nathan on 2022-12-26.
//
#include "../etcp.h"

int readn(SOCKET fd, char *bp, size_t len) {
    size_t cnt;
    ssize_t rc;
    cnt = len;
    while (cnt > 0) {
        rc = recv(fd, bp, cnt, 0);
        if (rc < 0) /* read error? */
        {
            if (errno == EINTR) /* interrupted? */
                continue; /* restart the read */
            return -1; /* return error */
        }
        if (rc == 0) /* EOF? */
            return (int)(len - cnt); /* return short count */
        bp += rc;
        cnt -= rc;
    }
    return (int) len;
}
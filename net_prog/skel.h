//
// Created by Nathan on 2022-12-25.
//

#ifndef DEV_QHC_CPP_PROJECTS_SKEL_H
#define DEV_QHC_CPP_PROJECTS_SKEL_H
/* UNIX version */

#define INIT() ( program_name = \
    strrchr( argv[ 0 ], '/' ) ) ? \
    program_name++ : \
    ( program_name = argv[ 0 ] )

#define EXIT(s) exit( s )

#define CLOSE(s) if ( close( s ) ) error( 1, errno, "closefailed" )

#define set_errno(e) errno = ( e )

#define isvalidsock(s) ( ( s ) >= 0 )

typedef int SOCKET;

#endif //DEV_QHC_CPP_PROJECTS_SKEL_H

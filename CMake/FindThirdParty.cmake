# Boost

# -DBoost_NO_BOOST_CMAKE=ON  -DBoost_NO_SYSTEM_PATHS=ON -DBOOST_ROOT=<local boost install>

# PCL looks for boost and affects the Boost_LIBRARIES variable so we call it first

FIND_PACKAGE(PCL)

set(Boost_ADDITIONAL_VERSIONS "1.49" "1.50.0" "1.51" "1.52" "1.53" "1.54" "1.55" "1.56" "1.57")
set(Boost_USE_STATIC_LIBS       OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

IF ( MSVC )
set(BOOST_COMPONENTS system filesystem regex serialization thread iostreams date_time chrono random program_options zlib)
ELSE ( MSVC )
set(BOOST_COMPONENTS system filesystem regex serialization thread iostreams date_time chrono random program_options)
ENDIF( MSVC )

FIND_PACKAGE(Boost 1.49 REQUIRED COMPONENTS ${BOOST_COMPONENTS})

SET(QT_COMPONENTS core gui network xml)

FIND_PACKAGE(Qt COMPONENTS ${QT_COMPONENTS})

FIND_PACKAGE(OpenCV 2.4 REQUIRED)
FIND_PACKAGE(Cairo REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)

IF ( NOT WIN32 )
FIND_PACKAGE(Freetype REQUIRED)
FIND_PACKAGE(Fontconfig REQUIRED)
ENDIF( NOT WIN32 )

FIND_PACKAGE(CURL)





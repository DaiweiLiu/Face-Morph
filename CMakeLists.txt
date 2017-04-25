cmake_minimum_required(VERSION 2.8.12)
project(face-morph)

include(dlib/cmake)

#   mkdir build
#   cmake ..
#   cmake --build . --config Release
#


find_package(OpenCV QUIET)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})

   add_executable(face_morph src/face_morph.cpp)
   target_link_libraries(face_morph dlib::dlib ${OpenCV_LIBS} )
else()
   message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
endif()




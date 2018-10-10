add_executable(train
        train.cpp
        rf.hpp
        rf.cpp
        dataset.hpp
        dataset.cpp
        obj20_dataset.hpp
        obj20_dataset.cpp
        certh_dataset.hpp
        certh_dataset.cpp
        util.cpp
        util.hpp
)
target_link_libraries(train cvx_util cvx_viz Qt5::Widgets ${OpenCV_LIBRARIES} )

add_executable(detect
        detect.cpp
        detector.cpp
        detector.hpp
        detector_impl.cpp
        detector_impl.hpp
        dataset.hpp
        dataset.cpp
        obj20_dataset.hpp
        obj20_dataset.cpp
        certh_dataset.hpp
        certh_dataset.cpp
        rf.hpp
        rf.cpp
        util.cpp
        util.hpp
        icp.hpp
        icp.cpp
)

target_link_libraries(detect cvx_util cvx_viz ${OpenCV_LIBRARIES})

add_executable(render_virtual_scene render_virtual_scene.cpp)
target_link_libraries(render_virtual_scene cvx_util cvx_viz)

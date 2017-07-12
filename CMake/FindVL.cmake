
SET(VL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/depends/vl/include/"  )

IF (WIN32)
	SET(VLCORE_LIBRARY_DEBUG "${CMAKE_SOURCE_DIR}/depends/vl/lib/vld.lib" )
	SET(VLCORE_LIBRARY_RELEASE "${CMAKE_SOURCE_DIR}/depends/vl/lib/vl.lib" )
	SET(VLNET_LIBRARY_DEBUG "${CMAKE_SOURCE_DIR}/depends/vl/lib/vlnetd.lib" )
	SET(VLNET_LIBRARY_RELEASE "${CMAKE_SOURCE_DIR}/depends/vl/lib/vlnet.lib" )

	SET(VLCORE_DLLS "${CMAKE_SOURCE_DIR}/depends/vl/bin/vld.dll" "${CMAKE_SOURCE_DIR}/depends/vl/bin/vl.dll")
	SET(VLNET_DLLS "${CMAKE_SOURCE_DIR}/depends/vl/bin/vlnetd.dll" "${CMAKE_SOURCE_DIR}/depends/vl/bin/vlnet.dll")
ELSE (WIN32)
	SET(VLCORE_LIBRARY_DEBUG "${CMAKE_SOURCE_DIR}/depends/vl/lib/libvld.so" )
	SET(VLCORE_LIBRARY_RELEASE "${CMAKE_SOURCE_DIR}/depends/vl/lib/libvl.so" )
	SET(VLNET_LIBRARY_DEBUG "${CMAKE_SOURCE_DIR}/depends/vl/lib/libvlnetd.so" )
	SET(VLNET_LIBRARY_RELEASE "${CMAKE_SOURCE_DIR}/depends/vl/lib/libvlnet.so" )
ENDIF (WIN32)

SET(VLCORE_LIBRARY debug ${VLCORE_LIBRARY_DEBUG} optimized ${VLCORE_LIBRARY_RELEASE})
SET(VLNET_LIBRARY debug ${VLNET_LIBRARY_DEBUG} optimized ${VLNET_LIBRARY_RELEASE})

MARK_AS_ADVANCED(VL_INCLUDE_DIR VL_CORE_LIBRARY VL_DLLS )
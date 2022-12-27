
add_mlir_dialect_library(TangentMLIR 
	TangentDialect.cpp
	TangentOps.cpp
	ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR/include/Tangent}


	LINK_LIBS PUBLIC
	MLIRIR
	)


#DEPENDS
# 	TANGENTMLIROpsIncGen

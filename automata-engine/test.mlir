// test.mlir
module {
  func.func @test_step(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> i32 {
    
    // Aquí está tu operación: (centro, izq, arriba, der, abajo)
    %res = automata.cell_update(%arg0, %arg1, %arg2, %arg3, %arg4) : (i32, i32, i32, i32, i32) -> i32
    
    return %res : i32
  }
}
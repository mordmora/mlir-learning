module {
  func.func @test_step(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> i32 {
    %res = automata.cell_update(%arg0, %arg1, %arg2, %arg3, %arg4) : (i32, i32, i32, i32, i32) -> i32
    return %res : i32
  }

  func.func @main() -> i32 {

    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    
    %res = func.call @test_step(%c0, %c1, %c1, %c1, %c0) : (i32, i32, i32, i32, i32) -> i32
    
    return %res : i32
  }
}
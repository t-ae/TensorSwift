import XCTest
@testable import TensorSwift

class TensorNNTest: XCTestCase {
    func testMaxPool() {
        let a = Tensor(shape: [2,2,3,1], elements: [0,1,2,3,4,5,6,7,8,9,10,11])
        var r = a.maxPool(ksize: [1,1,3,1], strides: [1,1,1,1])
        XCTAssertEqual(r, Tensor(shape: [2,2,3,1], elements: [1,2,2,4,5,5,7,8,8,10,11,11]))
        
        let b = Tensor(shape: [2,2,2,2], elements: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        r = b.maxPool(ksize:[1,1,2,1], strides: [1,1,1,1])
        XCTAssertEqual(r, Tensor(shape: [2,2,2,2], elements: [2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15]))
        
        r = b.maxPool(ksize:[1,1,2,1], strides: [1,1,2,1])
        XCTAssertEqual(r, Tensor(shape: [2,2,1,2], elements: [2, 3, 6, 7, 10, 11, 14, 15]))
    }
    
    func testConv2d() {
        XCTFail("Unimplemented yet.")
    }
}
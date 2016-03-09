import XCTest
@testable import TensorSwift

class TensorNNTest: XCTestCase {
    func testMaxPool() {
        XCTFail("Unimplemented yet.")
    }
    
    func testConv2d() {
        let a = Tensor(shape: [2,2,4,1], elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        let filter = Tensor(shape: [2,1,1,2], elements: [1,2,1,2])
        let result = a.conv2d(filter: filter, strides: [1,1,1,1])
        XCTAssertEqual(result, Tensor(shape: [2,2,4,2], elements: [6,12,8,16,10,20,12,24,5,10,6,12,7,14,8,16,22,44,24,48,26,52,28,56,13,26,14,28,15,30,16,32]))
    }
}
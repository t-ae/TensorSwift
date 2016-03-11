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
        let a = Tensor(shape: [2,2,4,1], elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        var filter = Tensor(shape: [2,1,1,2], elements: [1,2,1,2])
        var result = a.conv2d(filter: filter, strides: [1,1,1,1])
        XCTAssertEqual(result, Tensor(shape: [2,2,4,2], elements: [6,12,8,16,10,20,12,24,5,10,6,12,7,14,8,16,22,44,24,48,26,52,28,56,13,26,14,28,15,30,16,32]))
        
        filter = Tensor(shape: [1,1,1,5], elements: [1,2,1,2,3])
        result = a.conv2d(filter: filter, strides: [1,1,1,1])
        XCTAssertEqual(result, Tensor(shape: [2,2,4,5], elements: [1,2,1,2,3,2,4,2,4,6,3,6,3,6,9,4,8,4,8,12,5,10,5,10,15,6,12,6,12,18,7,14,7,14,21,8,16,8,16,24,9,18,9,18,27,10,20,10,20,30,11,22,11,22,33,12,24,12,24,36,13,26,13,26,39,14,28,14,28,42,15,30,15,30,45,16,32,16,32,48]))
        
        let b = Tensor(shape: [1,2,2,4], elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        filter = Tensor(shape: [1,1,4,2], elements: [1,2,1,2,3,2,1,1])
        result = b.conv2d(filter: filter, strides: [1,1,1,1])
        XCTAssertEqual(result, Tensor(shape: [1,2,2,2], elements: [16, 16, 40, 44, 64, 72, 88, 100]))
        
        let c = Tensor(shape: [1,4,2,2], elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        filter = Tensor(shape: [2,2,2,1], elements: [1,2,1,2,3,2,1,1])
        result = c.conv2d(filter: filter, strides: [1,2,2,1])
        XCTAssertEqual(result, Tensor(shape: [1,2,1,1], elements: [58,162]))
        
        let d = Tensor(shape: [1,4,4,1], elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        filter = Tensor(shape: [3,3,1,1], elements: [1,2,1,2,3,2,1,1,1])
        result = d.conv2d(filter: filter, strides: [1,3,3,1])
        XCTAssertEqual(result, Tensor(shape: [1,2,2,1], elements: [18,33,95,113]))
    }
}
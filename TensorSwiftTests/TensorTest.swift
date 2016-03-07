import XCTest
@testable import TensorSwift

class TensorTest: XCTestCase {
    func testIndex() {
        do {
            let a = Tensor(shape: [])
            XCTAssertEqual(a.index([]), 0)
        }
        
        do {
            let a = Tensor(shape: [7])
            XCTAssertEqual(a.index([3]), 3)
        }
        
        do {
            let a = Tensor(shape: [5, 7])
            XCTAssertEqual(a.index([1, 2]), 9)
        }
        
        do {
            let a = Tensor(shape: [5, 7, 11])
            XCTAssertEqual(a.index([3, 1, 2]), 244)
        }
    }
    
    func testMatmul() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [3, 4], elements: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
            let r = a.matmul(b)
            XCTAssertEqual(r, Tensor(shape: [2, 4], elements: [74, 80, 86, 92, 173, 188, 203, 218]))
        }
    }
}

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
    
    func testAdd() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [2, 3], elements: [7, 8, 9, 10, 11, 12])
            let r = a + b
            XCTAssertEqual(r, Tensor(shape: [2, 3], elements: [8, 10, 12, 14, 16, 18]))
        }
    }
    
    func testSub() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [2, 3], elements: [12, 11, 10, 9, 8, 7])
            let r = a - b
            XCTAssertEqual(r, Tensor(shape: [2, 3], elements: [-11, -9, -7, -5, -3, -1]))
        }
    }
    
    func testMul() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [2, 3], elements: [7, 8, 9, 10, 11, 12])
            let r = a * b
            XCTAssertEqual(r, Tensor(shape: [2, 3], elements: [7, 16, 27, 40, 55, 72]))
        }
    }
    
    func testDiv() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [2, 3], elements: [2, 4, 8, 16, 32, 64])
            let r = a / b
            XCTAssertEqual(r, Tensor(shape: [2, 3], elements: [0.5, 0.5, 0.375, 0.25, 0.15625, 0.09375]))
        }
    }
    
    func testMatmul() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [3, 4], elements: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
            let r = a.matmul(b)
            XCTAssertEqual(r, Tensor(shape: [2, 4], elements: [74, 80, 86, 92, 173, 188, 203, 218]))
        }
        do {
            let a = Tensor(shape: [3, 3], elements: [1, 1, 1, 2, 2, 2, 3, 3, 3])
            let b = Tensor(shape: [3, 3], elements: [1, 1, 1, 2, 2, 2, 3, 3, 3])
            let r = a.matmul(b)
            XCTAssertEqual(r, Tensor(shape: [3, 3], elements: [6, 6, 6, 12, 12, 12, 18, 18, 18]))
        }
    }
    
    func testMatmulPerformance(){
        let a = Tensor(shape: [1000, 1000], element: 0.1)
        let b = Tensor(shape: [1000, 1000], element: 0.1)
        measureBlock{
            a.matmul(b)
        }
    }
}

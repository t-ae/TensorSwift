
import XCTest
@testable import TensorSwift

class TensorSliceTests: XCTestCase {

    func testSlice() {
        let elem = (0..<3*4*5).map { Float($0) }
        let tensor = Tensor(shape: [3, 4, 5], elements: elem)
        
        do {
            let slice = tensor[0..<2, 1..<3, 1..<3]
            XCTAssertEqual(slice.shape, [2, 2, 2])
            XCTAssertEqual(slice.map { $0 }, [6.0, 7.0, 11.0, 12.0, 26.0, 27.0, 31.0, 32.0])
        }
        
        do {
            let slice = tensor[nil, nil, 0..<1]
            XCTAssertEqual(slice.shape, [3, 4, 1])
            XCTAssertEqual(slice.map { $0 }, [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
        }
    }
    
    func testSliceMap() {
        let elem = (0..<3*4*5).map { Float($0) }
        let tensor = Tensor(shape: [3, 4, 5], elements: elem)
        
        let slice = tensor[0..<2, 1..<3, 1..<3].map { $0*2 }
        XCTAssertEqual(slice, [6.0, 7.0, 11.0, 12.0, 26.0, 27.0, 31.0, 32.0].map { $0*2 })
    }
    
    func testSliceOfSlice() {
        let elem = (0..<3*4*5).map { Float($0) }
        let tensor = Tensor(shape: [3, 4, 5], elements: elem)
        
        let slice = tensor[0..<2, 1..<3, 1..<3]
        
        let sliceOfSlice = slice[0..<2, 1..<3, 1..<3]
        
        XCTAssertEqual(sliceOfSlice.shape, [2, 2, 2])
        XCTAssertEqual(sliceOfSlice.map{ $0 }, [6.0, 7.0, 11.0, 12.0, 26.0, 27.0, 31.0, 32.0])
    }
    
    func testMatmul() {
        do {
            let a = Tensor(shape: [3, 3], elements: (0..<3*3).map(Float.init))
            let b = Tensor(shape: [4, 4], elements: (0..<4*4).map(Float.init))
            
            let aSlice = a[1..<2, 1..<3] // [[4, 5]]
            let bSlice = b[1..<3, 1..<4] // [[5, 6, 7,] [9, 10, 11]]
            
            let r = aSlice.matmul(bSlice)
            XCTAssertEqual(r, Tensor(shape: [1, 3], elements: [65, 74, 83]))
        }
        do {
            let a = Tensor(shape: [3, 3], elements: [1, 1, 1, 2, 2, 2, 3, 3, 3])
            let b = Tensor(shape: [3, 3], elements: [1, 1, 1, 2, 2, 2, 3, 3, 3])
            
            let aSlice = a[0..<1, 0..<3]
            let bSlice = b[0..<3, 1..<2]
            
            let r = aSlice.matmul(bSlice)
            XCTAssertEqual(r, Tensor(shape: [1, 1], elements: [6]))
        }
    }
    
    func testMatmul_no_blas() {
        do {
            let a = Tensor(shape: [3, 3], elements: (0..<3*3).map(Float.init))
            let b = Tensor(shape: [4, 4], elements: (0..<4*4).map(Float.init))
            
            let aSlice = a[1..<2, 1..<3] // [[4, 5]]
            let bSlice = b[1..<3, 1..<4] // [[5, 6, 7,] [9, 10, 11]]
            
            let r = aSlice.matmul_no_blas(bSlice)
            XCTAssertEqual(r, Tensor(shape: [1, 3], elements: [65, 74, 83]))
        }
        do {
            let a = Tensor(shape: [3, 3], elements: [1, 1, 1, 2, 2, 2, 3, 3, 3])
            let b = Tensor(shape: [3, 3], elements: [1, 1, 1, 2, 2, 2, 3, 3, 3])
            
            let aSlice = a[0..<1, 0..<3]
            let bSlice = b[0..<3, 1..<2]
            
            let r = aSlice.matmul_no_blas(bSlice)
            XCTAssertEqual(r, Tensor(shape: [1, 1], elements: [6]))
        }
    }
    
    func testMaxPool() {
        do {
            let a = Tensor(shape: [3,3,1], elements: (0..<3*3*1).map(Float.init))
            let aSlice = a[0..<2, 1..<3, nil]
            let r = aSlice.maxPool(kernelSize: [2,2,1], strides: [2,2,1])
            XCTAssertEqual(r, Tensor(shape: [1,1,1], elements: [5]))
        }
        do {
            let a = Tensor(shape: [5,5,3], elements: (0..<5*5*3).map(Float.init))
            let aSlice = a[1..<3, 1..<4, 1..<3]
            let r = aSlice.maxPool(kernelSize: [2,2,1], strides: [1,1,1])
            XCTAssertEqual(r, Tensor(shape: [2,3,2], elements: [37,38,40,41,40,41,37,38,40,41,40,41]))
        }
    }
    
    static var allTests : [(String, (TensorSliceTests) -> () throws -> Void)] {
        return [
            ("testSlice", testSlice),
            ("testSliceMap", testSliceMap),
            ("testSliceOfSlice", testSliceOfSlice),
            ("testMatmul_no_blas", testMatmul_no_blas),
        ]
    }
}


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
    
    static var allTests : [(String, (TensorSliceTests) -> () throws -> Void)] {
        return [
            ("testSlice", testSlice),
            ("testSliceMap", testSliceMap),
            ("testSliceOfSlice", testSliceOfSlice),
        ]
    }
}

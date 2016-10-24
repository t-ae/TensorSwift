
import XCTest
import Foundation
@testable import TensorSwift

class NumpyTests: XCTestCase {

    func testLoadNpyA() {
        
        guard let x = load("a") else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [3])
        XCTAssertEqual(x.elements, [1.0, 2.0, 3.0])
    }
    
    func testLoadNpyB() {
        
        guard let x = load("b") else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [2, 2])
        XCTAssertEqual(x.elements, [1.0, 2.0, 3.0, 4.0])
    }
    
    func testLoadNpyC() {
        
        guard let x = load("c") else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [2, 2, 2])
        XCTAssertEqual(x.elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    }
    
    func testLoadNpyD() {
        
        guard let x = load("d") else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [4])
        XCTAssertEqual(x.elements, [1.0, 0.5, 0.25, 0.125])
    }
    
    func testLoadNpyE() {
    
        guard let x = load("e") else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [2, 3])
        XCTAssertEqual(x.elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }
    
    func load(_ fileName: String) -> Tensor? {
        guard let url = Bundle(for: type(of: self)).url(forResource: fileName, withExtension: "npy") else {
            return nil
        }
        
        guard let data = try? Data(contentsOf: url) else {
            return nil
        }
        
        return Tensor(npyData: data)
    }
}

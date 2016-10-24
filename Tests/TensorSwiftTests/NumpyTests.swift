
import XCTest
import Foundation
@testable import TensorSwift

class NumpyTests: XCTestCase {

    func testLoadNpyA() {
        
        
        guard let x = Tensor(npyData: NpyA) else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [3])
        XCTAssertEqual(x.elements, [1.0, 2.0, 3.0])
    }
    
    func testLoadNpyB() {
        
        guard let x = Tensor(npyData: NpyB) else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [2, 2])
        XCTAssertEqual(x.elements, [1.0, 2.0, 3.0, 4.0])
    }
    
    func testLoadNpyC() {
        
        guard let x = Tensor(npyData: NpyC) else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [2, 2, 2])
        XCTAssertEqual(x.elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    }
    
    func testLoadNpyD() {
        
        guard let x = Tensor(npyData: NpyD) else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [4])
        XCTAssertEqual(x.elements, [1.0, 0.5, 0.25, 0.125])
    }
    
    func testLoadNpyE() {
    
        guard let x = Tensor(npyData: NpyE) else {
            XCTFail()
            return
        }
        
        XCTAssertEqual(x.shape, [2, 3])
        XCTAssertEqual(x.elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }
    
    static var allTests : [(String, (NumpyTests) -> () throws -> Void)] {
        return [
            ("testLoadNpyA", testLoadNpyA),
            ("testLoadNpyB", testLoadNpyB),
            ("testLoadNpyC", testLoadNpyC),
            ("testLoadNpyD", testLoadNpyD),
            ("testLoadNpyE", testLoadNpyE),
        ]
    }

}

fileprivate let NpyA = Data(bytes:
    [UInt8](arrayLiteral:147, 78,85,77,80,89, 1, 0,70, 0,123,39,100,101,115,99,
            114,39,58,32,39,60,102,52,39,44,32,39,102,111,114,116,
            114,97,110,95,111,114,100,101,114,39,58,32,70,97,108,115,
            101,44,32,39,115,104,97,112,101,39,58,32,40,51,44,41,
            44,32,125,32,32,32,32,32,32,32,32,32,32,32,32,10,
            0,0,128,63, 0, 0, 0,64, 0, 0,64,64
    )
)

fileprivate let NpyB = Data(bytes:
    [UInt8](arrayLiteral: 147, 78, 85, 77, 80, 89,  1,  0, 70,  0,123, 39,100,101,115, 99,
            114, 39, 58, 32, 39, 60,102, 56, 39, 44, 32, 39,102,111,114,116,
            114, 97,110, 95,111,114,100,101,114, 39, 58, 32, 70, 97,108,115,
            101, 44, 32, 39,115,104, 97,112,101, 39, 58, 32, 40, 50, 44, 32,
            50, 41, 44, 32,125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10,
            0,  0,  0,  0,  0,  0,240, 63,  0,  0,  0,  0,  0,  0,  0, 64,
            0,  0,  0,  0,  0,  0,  8, 64,  0,  0,  0,  0,  0,  0, 16, 64
    )
)


fileprivate let NpyC = Data(bytes:
    [UInt8](arrayLiteral: 147, 78, 85, 77, 80, 89,  1,  0, 70,  0,123, 39,100,101,115, 99,
            114, 39, 58, 32, 39, 60,102, 52, 39, 44, 32, 39,102,111,114,116,
            114, 97,110, 95,111,114,100,101,114, 39, 58, 32, 70, 97,108,115,
            101, 44, 32, 39,115,104, 97,112,101, 39, 58, 32, 40, 50, 44, 32,
            50, 44, 32, 50, 41, 44, 32,125, 32, 32, 32, 32, 32, 32, 32, 10,
            0,  0,128, 63,  0,  0,  0, 64,  0,  0, 64, 64,  0,  0,128, 64,
            0,  0,160, 64,  0,  0,192, 64,  0,  0,224, 64,  0,  0,  0, 65
    )
)

fileprivate let NpyD = Data(bytes:
    [UInt8](arrayLiteral: 147, 78, 85, 77, 80, 89,  1,  0, 70,  0,123, 39,100,101,115, 99,
            114, 39, 58, 32, 39, 60,102, 56, 39, 44, 32, 39,102,111,114,116,
            114, 97,110, 95,111,114,100,101,114, 39, 58, 32, 70, 97,108,115,
            101, 44, 32, 39,115,104, 97,112,101, 39, 58, 32, 40, 52, 44, 41,
            44, 32,125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10,
            0,  0,  0,  0,  0,  0,240, 63,  0,  0,  0,  0,  0,  0,224, 63,
            0,  0,  0,  0,  0,  0,208, 63,  0,  0,  0,  0,  0,  0,192, 63
    )
)

fileprivate let NpyE = Data(bytes:
    [UInt8](arrayLiteral: 147, 78, 85, 77, 80, 89,  1,  0, 70,  0,123, 39,100,101,115, 99,
            114, 39, 58, 32, 39, 60,102, 52, 39, 44, 32, 39,102,111,114,116,
            114, 97,110, 95,111,114,100,101,114, 39, 58, 32, 70, 97,108,115,
            101, 44, 32, 39,115,104, 97,112,101, 39, 58, 32, 40, 50, 44, 32,
            51, 41, 44, 32,125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10,
            0,  0,128, 63,  0,  0,  0, 64,  0,  0, 64, 64,  0,  0,128, 64,
            0,  0,160, 64,  0,  0,192, 64
    )
)

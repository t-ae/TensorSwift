import XCTest
@testable import MNIST

class MNISTTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testDownloadTestData() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        let testData = downloadTestData()
        XCTAssertEqual(testData.images, NSData(contentsOfFile: "/Users/nisho/Documents/Qoncept/02_Project/Qoncept/tensorFlowTest/mnist/MNIST/MNIST/train-images-idx3-ubyte.data")!)
    }
    
    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }
    
}

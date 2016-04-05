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

        // original file size is 7840016, SHA-1 is "65e11ec1fd220343092a5070b58418b5c2644e26"
        XCTAssertEqual(testData.images.length, 7840016)
        XCTAssertEqual(testData.images.sha1(), "65e11ec1fd220343092a5070b58418b5c2644e26")
    }
    
    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measureBlock {
            // Put the code you want to measure the time of here.
        }
    }
    
}

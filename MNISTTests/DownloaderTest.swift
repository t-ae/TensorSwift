import XCTest

class Downloader: XCTestCase {
    func testDownloadTestData() {
        let testData = downloadTestData()
        
        // original file size is 7840016, SHA-1 is "65e11ec1fd220343092a5070b58418b5c2644e26"
        XCTAssertEqual(testData.images.length, 7840016)
        XCTAssertEqual(testData.images.sha1(), "65e11ec1fd220343092a5070b58418b5c2644e26")
    }
}

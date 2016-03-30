import Foundation
import zlib

func downloadTestData() -> (images: NSData, labels: NSData) {
    
    let sourceURL = "http://yann.lecun.com/exdb/mnist/"
    
    let test_images = NSData(contentsOfURL: NSURL(string: sourceURL + "t10k-images-idx3-ubyte.gz")!)!
    let test_labels = NSData(contentsOfURL: NSURL(string: sourceURL + "t10k-labels-idx1-ubyte.gz")!)!

    return (images: uncompressByGZip(test_images)!, labels: uncompressByGZip(test_labels)!)
}


func uncompressByGZip(source: NSData) -> NSData? {
    if source.length == 0 {
        return nil
    }
    
    var stream: z_stream = z_stream.init(next_in: UnsafeMutablePointer<Bytef>(source.bytes), avail_in: uint(source.length), total_in: 0, next_out: nil, avail_out: 0, total_out: 0, msg: nil, state: nil, zalloc: nil, zfree: nil, opaque: nil, data_type: 0, adler: 0, reserved: 0)
    if inflateInit2_(&stream, MAX_WBITS + 32, ZLIB_VERSION, Int32(sizeof(z_stream))) != Z_OK {
        return nil
    }
    
    let data = NSMutableData()
    
    while stream.avail_out == 0 {
        let buffer: UnsafeMutablePointer<Bytef> = UnsafeMutablePointer.alloc(0x10000)
        stream.next_out = buffer
        stream.avail_out = uint(sizeofValue(buffer))
        inflate(&stream, Z_FINISH)
        let length: size_t = sizeofValue(buffer) - Int(stream.avail_out)
        if length > 0 {
            data.appendBytes(buffer, length: length)
        }
    }
    
    inflateEnd(&stream)
    return data
}
// Npy file format
// https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
//

import Foundation

extension Tensor {
    
    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        try self.init(npyData: data)
    }
    
    public init(npyData: Data) throws {
        
        let magic = String(data: npyData.subdata(in: 0..<6), encoding: .ascii)
        guard magic == MAGIC_PREFIX else {
            throw NumpyError.ParseFailed(message: "Invalid prefix: \(magic)")
        }
        
        let major = npyData[6]
        guard major == 1 || major == 2 else {
            throw NumpyError.ParseFailed(message: "Invalid major version: \(major)")
        }
        
        let minor = npyData[7]
        guard minor == 0 else {
            throw NumpyError.ParseFailed(message: "Invalid minor version: \(minor)")
        }
        
        let headerLen: Int
        let rest: Data
        switch major {
        case 1:
            let tmp = Data(npyData[8...9]).withUnsafeBytes { (ptr: UnsafePointer<UInt8>) in
                ptr.withMemoryRebound(to: UInt16.self, capacity: 1) {
                    UInt16(littleEndian: $0.pointee)
                }
            }
            headerLen = Int(tmp)
            rest = npyData.subdata(in: 10..<npyData.count)
        case 2:
            let tmp = Data(npyData[8...11]).withUnsafeBytes { (ptr: UnsafePointer<UInt8>) in
                ptr.withMemoryRebound(to: UInt32.self, capacity: 1) {
                    UInt32(littleEndian: $0.pointee)
                }
            }
            headerLen = Int(tmp)
            rest = npyData.subdata(in: 12..<npyData.count)
        default:
            fatalError("Never happens.")
        }
        
        let headerData = rest.subdata(in: 0..<headerLen)
        let header = try parseHeader(headerData)
        
        let elemCount = header.shape.volume()
        let elemData = rest.subdata(in: headerLen..<rest.count)
        let elements: [Float]
        
        switch (header.dataType, header.isLittleEndian) {
        case (.float32, true):
            elements = elemData.withUnsafeBytes { (ptr: UnsafePointer<UInt8>) in
                ptr.withMemoryRebound(to: Float32.self, capacity: elemCount) { ptr2 in
                    (0..<elemCount).map { Float(ptr2.advanced(by: $0).pointee) }
                }
            }
        case (.float64, true):
            elements = elemData.withUnsafeBytes { (ptr: UnsafePointer<UInt8>) in
                ptr.withMemoryRebound(to: Float64.self, capacity: elemCount) { ptr2 in
                    (0..<elemCount).map { Float(ptr2.advanced(by: $0).pointee) }
                }
            }
        case (.float32, false):
            let uints = elemData.withUnsafeBytes { (ptr: UnsafePointer<UInt8>) in
                ptr.withMemoryRebound(to: UInt32.self, capacity: elemCount) { ptr2 in
                    (0..<elemCount).map { UInt32(bigEndian: ptr2.advanced(by: $0).pointee) }
                }
            }
            elements = uints.map { Float(Float32(bitPattern: $0)) }
        case (.float64, false):
            let uints = elemData.withUnsafeBytes { (ptr: UnsafePointer<UInt8>) in
                ptr.withMemoryRebound(to: UInt64.self, capacity: elemCount) { ptr2 in
                    (0..<elemCount).map { UInt64(bigEndian: ptr2.advanced(by: $0).pointee) }
                }
            }
            elements = uints.map { Float(Float64(bitPattern: $0)) }
        }
        
        self.init(shape: header.shape, elements: elements)
    }
}

public enum NumpyError: Error {
    case ParseFailed(message: String)
}

private let MAGIC_PREFIX = "\u{93}NUMPY"

private struct NumpyHeader {
    let shape: Shape
    let dataType: DataType
    let isLittleEndian: Bool
    let isFortranOrder: Bool
    let descr: String
}

private func parseHeader(_ data: Data) throws -> NumpyHeader {
    
    guard let str = String(data: data, encoding: .ascii) else {
        throw NumpyError.ParseFailed(message: "Failed to load header")
    }
    
    let descr: String
    let isLittleEndian: Bool
    let dataType: DataType
    let isFortranOrder: Bool
    do {
        let separate = str.components(separatedBy: CharacterSet(charactersIn: ", ")).filter { !$0.isEmpty }
        
        guard let descrIndex = separate.index(where: { $0.contains("descr") }) else {
            throw NumpyError.ParseFailed(message: "Header does not contain the key 'descr'")
        }
        descr = separate[descrIndex + 1]
        
        isLittleEndian = descr.contains("<") || descr.contains("|")
        
        guard let dt = DataType.all.filter({ descr.contains($0.rawValue) }).first else {
            fatalError("Unsupported dtype: \(descr)")
        }
        dataType = dt
        
        guard let fortranIndex = separate.index(where: { $0.contains("fortran_order") }) else {
            throw NumpyError.ParseFailed(message: "Header does not contain the key 'fortran_order'")
        }
        
        isFortranOrder = separate[fortranIndex+1].contains("True")
        
        guard !isFortranOrder else {
            fatalError("\"fortran_order\" must be False.")
        }
    }
    
    let shape: Shape
    do {
        guard let left = str.range(of: "("),
            let right = str.range(of: ")") else {
                throw NumpyError.ParseFailed(message: "Shape not found in header.")
        }
        
        let substr = str.substring(with: left.upperBound..<right.lowerBound)
        var dimens = [Dimension]()
        
        let strs = substr.replacingOccurrences(of: " ", with: "")
            .components(separatedBy: ",")
            .filter { !$0.isEmpty }
        for s in strs {
            guard let i = Int(s) else {
                throw NumpyError.ParseFailed(message: "Shape contains invalid integer: \(s)")
            }
            dimens.append(Dimension(i))
        }
        shape = Shape(dimens)
    }
    
    return NumpyHeader(shape: shape,
                       dataType: dataType,
                       isLittleEndian: isLittleEndian,
                       isFortranOrder: isFortranOrder,
                       descr: descr)
}

private enum DataType: String {
    case float32 = "f4"
    case float64 = "f8"
    
    static var all: [DataType] {
        return [.float32, .float64]
    }
}

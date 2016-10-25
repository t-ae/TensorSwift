// Npy file format
// https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
//

import Foundation

extension Tensor {
    
    public init?(contentsOf url: URL) {
        guard let data = try? Data(contentsOf: url) else {
            return nil
        }
        self.init(npyData: data)
    }
    
    public init?(npyData: Data) {
        
        let magic = String(data: npyData.subdata(in: 0..<6), encoding: .ascii)
        guard magic == MAGIC_PREFIX else {
            return nil
        }
        
        let major = npyData[6]
        guard major == 1 || major == 2 else {
            return nil
        }
        
        let minor = npyData[7]
        guard minor == 0 else {
            return nil
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
        guard let header = parseHeader(headerData) else {
            return nil
        }
        
        guard header.isLittleEndian else {
            return nil
        }
        
        guard !header.isFortranOrder else {
            return nil
        }
        
        let elemCount = header.shape.volume()
        let elemData = rest.subdata(in: headerLen..<rest.count)
        let elements: [Float]
        
        switch header.dataType {
        case .Float32:
            elements = elemData.withUnsafeBytes { (ptr: UnsafePointer<UInt8>) in
                ptr.withMemoryRebound(to: Float32.self, capacity: elemCount) { ptr2 -> [Float] in
                    (0..<elemCount).map { Float32(ptr2.advanced(by: $0).pointee) }
                }
            }
        case .Float64:
            elements = elemData.withUnsafeBytes { (ptr: UnsafePointer<UInt8>) in
                ptr.withMemoryRebound(to: Float64.self, capacity: elemCount) { ptr2 in
                    (0..<elemCount).map { Float(ptr2.advanced(by: $0).pointee) }
                }
            }
        }
        
        self.init(shape: header.shape, elements: elements)
    }
}

private let MAGIC_PREFIX = "\u{93}NUMPY"

private struct NumpyHeader {
    let shape: Shape
    let dataType: DataType
    let isLittleEndian: Bool
    let isFortranOrder: Bool
}

private func parseHeader(_ data: Data) -> NumpyHeader? {
    
    guard let str = String(data: data, encoding: .ascii) else {
        return nil
    }
    
    let isLittleEndian: Bool
    let dataType: DataType
    let isFortranOrder: Bool
    do {
        let separate = str.components(separatedBy: CharacterSet(charactersIn: ", ")).filter { !$0.isEmpty }
        
        guard let descrIndex = separate.index(where: { $0.contains("descr") }) else {
            return nil
        }
        let descr = separate[descrIndex + 1]
        
        isLittleEndian = descr.contains("<") || descr.contains("|")
        
        guard let dt = DataType.all.filter({ descr.contains($0.rawValue) }).first else {
            return nil
        }
        dataType = dt
        
        guard let fortranIndex = separate.index(where: { $0.contains("fortran_order") }) else {
            return nil
        }
        
        isFortranOrder = separate[fortranIndex+1].contains("True")
    }
    
    let shape: Shape
    do {
        guard let left = str.range(of: "("),
            let right = str.range(of: ")") else {
                return nil
        }
        
        let substr = str.substring(with: left.upperBound..<right.lowerBound)
        var dimens = [Dimension]()
        
        let strs = substr.replacingOccurrences(of: " ", with: "")
            .components(separatedBy: ",")
            .filter { !$0.isEmpty }
        for s in strs {
            guard let i = Int(s) else {
                return nil
            }
            dimens.append(Dimension(i))
        }
        shape = Shape(dimens)
    }
    
    return NumpyHeader(shape: shape,
                       dataType: dataType,
                       isLittleEndian: isLittleEndian,
                       isFortranOrder: isFortranOrder)
}

private enum DataType: String {
    case Float32 = "f4"
    case Float64 = "f8"
    
    static var all: [DataType] {
        return [.Float32, .Float64]
    }
}

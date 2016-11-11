
import Accelerate

public protocol TensorProtocol : Sequence {
    typealias Element = Float
    
    var shape: Shape { get }
    
    func map(_ transform: (Element) throws ->Element) rethrows -> [Element]
}

extension TensorProtocol {
    internal var wholeElements: [Element] {
        switch self {
        case is Tensor:
            return (self as! Tensor).elements
        case is TensorSlice:
            return (self as! TensorSlice).tensor.elements
        default:
            fatalError("Unsupported type: \(type(of: self))")
        }
    }
    
    internal var wholeRanges: [CountableRange<Int>] {
        switch self {
        case is Tensor:
            return (self as! Tensor).shape.dimensions.map { 0..<$0.value }
        case is TensorSlice:
            return (self as! TensorSlice).tensor.wholeRanges
        default:
            fatalError("Unsupported type: \(type(of: self))")
        }
    }
    
    internal var ranges: [CountableRange<Int>] {
        switch self {
        case is Tensor:
            return (self as! Tensor).wholeRanges
        case is TensorSlice:
            return (self as! TensorSlice)._ranges
        default:
            fatalError("Unsupported type: \(type(of: self))")
        }
    }
}

internal func commutativeBinaryOperation<A: TensorProtocol, B: TensorProtocol>(_ lhs: A, _ rhs: B, operation: (Float, Float)-> Float) -> Tensor
    where A.Iterator.Element==Float, B.Iterator.Element==Float {
        
        let lSize = lhs.shape.dimensions.count
        let rSize = rhs.shape.dimensions.count
        
        if lSize == rSize {
            precondition(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
            return Tensor(shape: lhs.shape, elements: zip(lhs,rhs).map(operation))
        }
        
        if lSize < rSize {
            assert(hasSuffix(array: rhs.shape.dimensions, suffix: lhs.shape.dimensions), "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
            return Tensor(shape: rhs.shape, elements: zip(rhs, RepeatedSequence(lhs)).map(operation))
        } else {
            assert(hasSuffix(array: lhs.shape.dimensions, suffix: rhs.shape.dimensions), "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
            return Tensor(shape: lhs.shape, elements: zip(lhs, RepeatedSequence(rhs)).map(operation))
        }
}

internal func noncommutativeBinaryOperation<A: TensorProtocol, B: TensorProtocol>(_ lhs: A, _ rhs: B, operation: (Float, Float)-> Float) -> Tensor
    where A.Iterator.Element==Float, B.Iterator.Element==Float {
        let lSize = lhs.shape.dimensions.count
        let rSize = rhs.shape.dimensions.count
        
        if lSize == rSize {
            precondition(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
            return Tensor(shape: lhs.shape, elements: zip(lhs, rhs).map(operation))
        } else if lSize < rSize {
            precondition(hasSuffix(array: rhs.shape.dimensions, suffix: lhs.shape.dimensions), "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
            return Tensor(shape: rhs.shape, elements: zip(rhs, RepeatedSequence(lhs)).map{ operation($1, $0) })
        } else {
            precondition(hasSuffix(array: lhs.shape.dimensions, suffix: rhs.shape.dimensions), "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
            return Tensor(shape: lhs.shape, elements: zip(lhs, RepeatedSequence(rhs)).map(operation))
        }
}

public func +<A: TensorProtocol, B: TensorProtocol>(lhs: A, rhs: B) -> Tensor
    where A.Iterator.Element==Float, B.Iterator.Element==Float {
        return commutativeBinaryOperation(lhs, rhs, operation: +)
}

public func -<A: TensorProtocol, B: TensorProtocol>(lhs: A, rhs: B) -> Tensor
    where A.Iterator.Element==Float, B.Iterator.Element==Float {
        return noncommutativeBinaryOperation(lhs, rhs, operation: -)
}

public func *<A: TensorProtocol, B: TensorProtocol>(lhs: A, rhs: B) -> Tensor
    where A.Iterator.Element==Float, B.Iterator.Element==Float {
        return commutativeBinaryOperation(lhs, rhs, operation: *)
}

public func /<A: TensorProtocol, B: TensorProtocol>(lhs: A, rhs: B) -> Tensor
    where A.Iterator.Element==Float, B.Iterator.Element==Float {
        return noncommutativeBinaryOperation(lhs, rhs, operation: /)
}

public func *<T: TensorProtocol>(lhs: T, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.map { $0 * rhs })
}

public func *<T: TensorProtocol>(lhs: Float, rhs: T) -> Tensor {
    return Tensor(shape: rhs.shape, elements: rhs.map { lhs * $0 })
}

public func /<T: TensorProtocol>(lhs: T, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.map { $0 / rhs })
}

public func /<T: TensorProtocol>(lhs: Float, rhs: T) -> Tensor {
    return Tensor(shape: rhs.shape, elements: rhs.map { lhs / $0 })
}

extension TensorProtocol { // Matrix
    public func matmul<T: TensorProtocol>(_ tensor: T) -> Tensor {
        precondition(shape.dimensions.count == 2, "This tensor is not a matrix: shape = \(shape)")
        precondition(tensor.shape.dimensions.count == 2, "The given tensor is not a matrix: shape = \(tensor.shape)")
        precondition(tensor.shape.dimensions[0] == shape.dimensions[1], "Incompatible shapes of matrices: self.shape = \(shape), tensor.shape = \(tensor.shape)")
        
        #if os(iOS) || os(OSX)
            let result = Tensor(shape: [shape.dimensions[0], tensor.shape.dimensions[1]])
            
            let wholeRanges = (a: self.wholeRanges, b: tensor.wholeRanges)
            let ranges = (a: self.ranges, b: tensor.ranges)
            
            let m = Int32(shape.dimensions[0].value)
            let n = Int32(tensor.shape.dimensions[1].value)
            let k = Int32(shape.dimensions[1].value)
            
            let lda = wholeRanges.a[1].count
            let ldb = wholeRanges.b[1].count
            
            let offsetA = lda*ranges.a[0].lowerBound+ranges.a[1].lowerBound
            let offsetB = ldb*ranges.b[0].lowerBound+ranges.b[1].lowerBound

            let aPtr = UnsafeMutablePointer<Float>(mutating: self.wholeElements).advanced(by: offsetA)
            
            var elemsB: [Float] = tensor.wholeElements
            let bPtr = UnsafeMutablePointer<Float>(&elemsB).advanced(by: offsetB)
            
            cblas_sgemm(
                CblasRowMajor,                                // Order
                CblasNoTrans,                                 // TransA
                CblasNoTrans,                                 // TransB
                m,                                            // M
                n,                                            // N
                k,                                            // K
                1.0,                                          // alpha
                aPtr,                                         // A
                Int32(lda),                                   // lda
                bPtr,                                         // B
                Int32(ldb),                                   // ldb
                1.0,                                          // beta
                UnsafeMutablePointer<Float>(mutating: result.elements), // C
                n                                             // ldc
            )
            
            return result
        #else
            return matmul_no_blas(tensor)
        #endif
    }
    
    internal func matmul_no_blas<T: TensorProtocol>(_ tensor: T) -> Tensor {
        let n = shape.dimensions[1].value
        
        let numRows = shape.dimensions[0]
        let numCols = tensor.shape.dimensions[1]
        
        let wholeRanges = (a: self.wholeRanges, b: tensor.wholeRanges)
        let ranges = (a: self.ranges, b: tensor.ranges)
        
        let lda = wholeRanges.a[1].count
        let ldb = wholeRanges.b[1].count
        
        let offsetA = lda*ranges.a[0].lowerBound+ranges.a[1].lowerBound
        let offsetB = ldb*ranges.b[0].lowerBound+ranges.b[1].lowerBound
        
        let leftHead = UnsafeMutablePointer<Float>(mutating: self.wholeElements).advanced(by: offsetA)
        let rightHead = UnsafeMutablePointer<Float>(mutating: tensor.wholeElements).advanced(by: offsetB)
        
        let elements = [Float](repeating: 0.0, count: (numCols * numRows).value)
        for r in 0..<numRows.value {
            for i in 0..<n {
                var pointer = UnsafeMutablePointer<Float>(mutating: elements) + r * numCols.value
                let left = leftHead[r * lda + i]
                var rightPointer = rightHead + i * ldb
                for _ in 0..<numCols.value {
                    pointer.pointee += left * rightPointer.pointee
                    pointer += 1
                    rightPointer += 1
                }
            }
        }
        
        return Tensor(shape: [numRows, numCols], elements: elements)
    }
}

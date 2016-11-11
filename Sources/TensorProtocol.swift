
import Accelerate

public protocol TensorProtocol : Sequence {
    typealias Element = Float
    
    var shape: Shape { get }
    
    func map(_ transform: (Element) throws ->Element) rethrows -> [Element]
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

extension Tensor { // Matrix
    public func matmul(_ tensor: Tensor) -> Tensor {
        precondition(shape.dimensions.count == 2, "This tensor is not a matrix: shape = \(shape)")
        precondition(tensor.shape.dimensions.count == 2, "The given tensor is not a matrix: shape = \(tensor.shape)")
        precondition(tensor.shape.dimensions[0] == shape.dimensions[1], "Incompatible shapes of matrices: self.shape = \(shape), tensor.shape = \(tensor.shape)")
        
        #if os(iOS) || os(OSX)
            let result = Tensor(shape: [shape.dimensions[0], tensor.shape.dimensions[1]])
            
            let n = Int32(tensor.shape.dimensions[1].value)
            let k = Int32(shape.dimensions[1].value)
            cblas_sgemm(
                CblasRowMajor,                                // Order
                CblasNoTrans,                                 // TransA
                CblasNoTrans,                                 // TransB
                Int32(shape.dimensions[0].value),             // M
                n,                                            // N
                k,                                            // K
                1.0,                                          // alpha
                elements,                                     // A
                k,                                            // lda
                tensor.elements,                              // B
                n,                                            // ldb
                1.0,                                          // beta
                UnsafeMutablePointer<Float>(mutating: result.elements), // C
                n                                             // ldc
            )
            
            return result
        #else
            let n = shape.dimensions[1].value
            
            let numRows = shape.dimensions[0]
            let numCols = tensor.shape.dimensions[1]
            
            let leftHead = UnsafeMutablePointer<Float>(self.elements)
            let rightHead = UnsafeMutablePointer<Float>(tensor.elements)
            
            let elements = [Float](count: (numCols * numRows).value, repeatedValue: 0.0)
            for r in 0..<numRows.value {
                for i in 0..<n {
                    var pointer = UnsafeMutablePointer<Float>(elements) + r * numCols.value
                    let left = leftHead[r * n + i]
                    var rightPointer = rightHead + i * numCols.value
                    for _ in 0..<numCols.value {
                        pointer.memory += left * rightPointer.memory
                        pointer += 1
                        rightPointer += 1
                    }
                }
            }
            
            return Tensor(shape: [numRows, numCols], elements: elements)
        #endif
    }
}

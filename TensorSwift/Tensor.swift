
import Accelerate

public struct Tensor {
    public typealias Element = Float
    
    public let shape: Shape
    public private(set) var elements: [Element]
    
    public init(shape: Shape, elements: [Element]) {
        let c = shape.volume
        assert(elements.count >= c, "`elements.count` must be greater than or equal to `shape.volume`: elements.count = \(elements.count), shape.volume = \(shape.volume)")
        self.shape = shape
        self.elements = (elements.count == c) ? elements : Array(elements[0..<c])
    }
}

extension Tensor { // Additional Initializers
    public init(shape: Shape, element: Element = 0.0) {
        self.init(shape: shape, elements: [Element](count: shape.volume, repeatedValue: element))
    }
}

extension Tensor {
    public func reshape(shape: Shape) -> Tensor {
        return Tensor(shape: shape, elements: elements)
    }
}

extension Tensor { // like CollentionType
    internal func index(indices: [Int]) -> Int {
        assert(indices.count == shape.dimensions.count, "`indices.count` must be \(shape.dimensions.count): \(indices.count)")
        return zip(shape.dimensions, indices).reduce(0) {
            assert(0 <= $1.1 && $1.1 < $1.0.value, "Illegal index: indices = \(indices), shape = \(shape)")
            return $0 * $1.0.value + $1.1
        }
    }
    
    public subscript(indices: Int...) -> Element {
        get {
            return elements[index(indices)]
        }
        set {
            elements[index(indices)] = newValue
        }
    }
    
    public var volume: Int {
        return shape.volume
    }
}

extension Tensor: SequenceType {
    public func generate() -> IndexingGenerator<[Element]> {
        return elements.generate()
    }
}

extension Tensor: Equatable {}
public func ==(lhs: Tensor, rhs: Tensor) -> Bool {
    return lhs.shape == rhs.shape && lhs.elements == rhs.elements
}

private func commutativeBinaryOperation(lhs: Tensor, _ rhs: Tensor, operation: (Float, Float) -> Float) -> Tensor {
    let lSize = lhs.shape.dimensions.count
    let rSize = rhs.shape.dimensions.count
    
    if lSize == rSize {
        assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}")
        return Tensor(shape: lhs.shape, elements: zipMap(lhs.elements, rhs.elements, operation: operation))
    }
    
    let a: Tensor
    let b: Tensor
    if lSize < rSize {
        a = rhs
        b = lhs
    } else {
        a = lhs
        b = rhs
    }
    assert(hasSuffix(array: a.shape.dimensions, suffix: b.shape.dimensions), "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}")
    
    return Tensor(shape: a.shape, elements: zipMapRepeat(a.elements, b.elements, operation: operation))
}

private func noncommutativeBinaryOperation(lhs: Tensor, _ rhs: Tensor, operation: (Float, Float) -> Float) -> Tensor {
    let lSize = lhs.shape.dimensions.count
    let rSize = rhs.shape.dimensions.count
    
    if lSize == rSize {
        assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}")
        return Tensor(shape: lhs.shape, elements: zipMap(lhs.elements, rhs.elements, operation: operation))
    } else if lSize < rSize {
        assert(hasSuffix(array: rhs.shape.dimensions, suffix: lhs.shape.dimensions), "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}")
        return Tensor(shape: rhs.shape, elements: zipMapRepeat(rhs.elements, lhs.elements, operation: { operation($1, $0) }))
    } else {
        assert(hasSuffix(array: lhs.shape.dimensions, suffix: rhs.shape.dimensions), "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}")
        return Tensor(shape: lhs.shape, elements: zipMapRepeat(lhs.elements, rhs.elements, operation: operation))
    }
}

public func +(lhs: Tensor, rhs: Tensor) -> Tensor {
    return commutativeBinaryOperation(lhs, rhs, operation: +)
}

public func -(lhs: Tensor, rhs: Tensor) -> Tensor {
    return noncommutativeBinaryOperation(lhs, rhs, operation: -)
}

public func *(lhs: Tensor, rhs: Tensor) -> Tensor {
    return commutativeBinaryOperation(lhs, rhs, operation: *)
}

public func /(lhs: Tensor, rhs: Tensor) -> Tensor {
    return noncommutativeBinaryOperation(lhs, rhs, operation: /)
}

public func *(lhs: Tensor, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.elements.map { $0 * rhs })
}

public func *(lhs: Float, rhs: Tensor) -> Tensor {
    return Tensor(shape: rhs.shape, elements: rhs.elements.map { lhs * $0 })
}

public func /(lhs: Tensor, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.elements.map { $0 / rhs })
}

public func /(lhs: Float, rhs: Tensor) -> Tensor {
    return Tensor(shape: rhs.shape, elements: rhs.elements.map { lhs / $0 })
}

extension Tensor { // Matrix
    public func matmul(tensor: Tensor) -> Tensor {
        assert(shape.dimensions.count == 2, "This tensor is not a matrix: shape = \(shape)")
        assert(tensor.shape.dimensions.count == 2, "The given tensor is not a matrix: shape = \(tensor.shape)")
        assert(tensor.shape.dimensions[0] == shape.dimensions[1], "Incompatible shapes of matrices: self.shape = \(shape), tensor.shape = \(tensor.shape)")
        
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
            UnsafeMutablePointer<Float>(result.elements), // C
            n                                             // ldc
        )
        
        return result
    }
}

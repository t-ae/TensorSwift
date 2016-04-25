
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
    assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
    return lhs.elements == rhs.elements
}

public func +(lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
    return Tensor(shape: lhs.shape, elements: zip(lhs.elements, rhs.elements).map(+))
}

public func -(lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
    return Tensor(shape: lhs.shape, elements: zip(lhs.elements, rhs.elements).map(-))
}

public func *(lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
    return Tensor(shape: lhs.shape, elements: zip(lhs.elements, rhs.elements).map(*))
}

public func /(lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
    return Tensor(shape: lhs.shape, elements: zip(lhs.elements, rhs.elements).map(/))
}

public func *(lhs: Tensor, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.elements.map { $0 * rhs })
}

public func *(lhs: Float, rhs: Tensor) -> Tensor {
    return Tensor(shape: rhs.shape, elements: rhs.elements.map { $0 * lhs })
}

public func /(lhs: Tensor, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.elements.map { $0 / rhs })
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

public struct Tensor {
    public typealias Element = Float
    
    public let shape: Shape
    public private(set) var elements: [Element]
    
    public init(shape: Shape, elements: [Element]) {
        let c = shape.count
        assert(elements.count >= c, "`elements.count` must be greater than or equal to `shape.count`: elements.count = \(elements.count), shape.count = \(shape.count)")
        self.shape = shape
        self.elements = (elements.count == c) ? elements : Array(elements[0..<c])
    }
}

extension Tensor { // Additional Initializers
    public init(shape: Shape, element: Element = 0.0) {
        self.init(shape: shape, elements: [Element](count: shape.count, repeatedValue: element))
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
    
    public var count: Int {
        return shape.count
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
        
        let n = shape.dimensions[1]
        assert(tensor.shape.dimensions[0] == n, "Incompatible shapes of matrices: self.shape = \(shape), tensor.shape = \(tensor.shape)")
        
        let numRows = shape.dimensions[0]
        let numCols = tensor.shape.dimensions[1]
        
        var elements: [Element] = []
        elements.reserveCapacity(numCols.value * numRows.value)
        for r in 0..<numRows.value {
            for c in 0..<numCols.value {
                var e: Element = 0.0
                for i in 0..<n.value {
                    e += self[r, i] * tensor[i, c]
                }
                elements.append(e)
            }
        }
        
        return Tensor(shape: [numRows, numCols], elements: elements)
    }
}

extension Tensor { // Matrix
    public func matmul_fast(tensor: Tensor) -> Tensor {
        // matmulの高速化
        assert(shape.dimensions.count == 2, "This tensor is not a matrix: shape = \(shape)")
        assert(tensor.shape.dimensions.count == 2, "The given tensor is not a matrix: shape = \(tensor.shape)")
        
        let n = shape.dimensions[1]
        assert(tensor.shape.dimensions[0] == n, "Incompatible shapes of matrices: self.shape = \(shape), tensor.shape = \(tensor.shape)")
        
        let numRows = shape.dimensions[0]
        let numCols = tensor.shape.dimensions[1]
        
        var elements: [Element] = [Element](count: numCols.value * numRows.value, repeatedValue: 0)
        for r in 0..<numRows.value {
            for i in 0..<n.value {
                let tmp = self.elements[r * n.value + i]
                for c in 0..<numCols.value {
                    elements[r * numCols.value + c] += tmp * tensor.elements[i * numCols.value + c]
                }
            }
        }
        
        return Tensor(shape: [numRows, numCols], elements: elements)
    }
}

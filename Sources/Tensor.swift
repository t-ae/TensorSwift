
#if os(iOS) || os(OSX)
import Accelerate
#endif

public struct Tensor : TensorProtocol {
    
    public typealias Element = Float
    
    public let shape: Shape
    public fileprivate(set) var elements: [Element]
    
    public init(shape: Shape, elements: [Element]) {
        let volume = shape.volume()
        precondition(elements.count >= volume, "`elements.count` must be greater than or equal to `shape.volume`: elements.count = \(elements.count), shape.volume = \(shape.volume())")
        self.shape = shape
        self.elements = (elements.count == volume) ? elements : Array(elements[0..<volume])
    }
}

extension Tensor { // Additional Initializers
    public init(shape: Shape, element: Element = 0.0) {
        self.init(shape: shape, elements: [Element](repeating: element, count: shape.volume()))
    }
}

extension Tensor {
    public init(slice: TensorSlice) {
        self.init(shape: slice.shape, elements: slice.map { $0 })
    }
}

extension Tensor {
    public mutating func reshape(_ shape: Shape) {
        self = reshaped(shape)
    }
    
    public func reshaped(_ shape: Shape) -> Tensor {
        return Tensor(shape: shape, elements: elements)
    }
}

extension Tensor { // like CollentionType
    internal func index(_ indices: [Int]) -> Int {
        assert(indices.count == shape.dimensions.count, "`indices.count` must be \(shape.dimensions.count): \(indices.count)")
        return zip(shape.dimensions, indices).reduce(0) {
            assert(0 <= $1.1 && $1.1 < $1.0.value, "Illegal index: indices = \(indices), shape = \(shape)")
            return $0 * $1.0.value + $1.1
        }
    }
    
    public subscript(indices: Int...) -> Element {
        get {
            return _subscript(indices)
        }
        set {
            elements[index(indices)] = newValue
        }
    }
    
    internal func _subscript(_ indices: [Int]) -> Element {
        return elements[index(indices)]
    }
    
    public func volume() -> Int {
        return shape.volume()
    }
}

extension Tensor {
    
    internal func _subscript(ranges: [CountableRange<Int>]) -> TensorSlice {
        precondition(ranges.count == self.shape.dimensions.count)
        return TensorSlice(tensor: self, ranges: ranges)
    }
    
    public subscript(ranges: Range<Int>...) -> TensorSlice {
        return _subscript(ranges: ranges.map { CountableRange($0) })
    }
    
    public subscript(ranges: ClosedRange<Int>...) -> TensorSlice {
        return _subscript(ranges: ranges.map { CountableRange($0) })
    }
    
    public subscript(ranges: CountableRange<Int>...) -> TensorSlice {
        return _subscript(ranges: ranges)
    }
    
    public subscript(ranges: CountableClosedRange<Int>...) -> TensorSlice {
        return _subscript(ranges: ranges.map { CountableRange($0) })
    }
    
    public subscript(ranges: Range<Int>?...) -> TensorSlice {
        let validRanges = ranges.enumerated().map { i, range in
            range.map(CountableRange.init) ?? 0..<self.shape.dimensions[i].value
        }
        return _subscript(ranges: validRanges)
    }
    
    public subscript(ranges: ClosedRange<Int>?...) -> TensorSlice {
        let validRanges = ranges.enumerated().map { i, range in
            range.map(CountableRange.init) ?? 0..<self.shape.dimensions[i].value
        }
        return _subscript(ranges: validRanges)
    }
    
    public subscript(ranges: CountableRange<Int>?...) -> TensorSlice {
        let validRanges = ranges.enumerated().map { i, range in
            range ?? 0..<self.shape.dimensions[i].value
        }
        return _subscript(ranges: validRanges)
    }
    
    public subscript(ranges: CountableClosedRange<Int>?...) -> TensorSlice {
        let validRanges = ranges.enumerated().map { i, range in
            range.map(CountableRange.init) ?? 0..<self.shape.dimensions[i].value
        }
        return _subscript(ranges: validRanges)
    }
}

extension Tensor: Sequence {
    public func makeIterator() -> IndexingIterator<[Element]> {
        return elements.makeIterator()
    }
}

extension Tensor: Equatable {}
public func ==(lhs: Tensor, rhs: Tensor) -> Bool {
    return lhs.shape == rhs.shape && lhs.elements == rhs.elements
}

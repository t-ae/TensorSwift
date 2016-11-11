

public struct TensorSlice : TensorProtocol {

    public typealias Element = Float
    
    internal let tensor: Tensor
    internal let _ranges: [CountableRange<Int>]
    
    public init(tensor: Tensor, ranges: [CountableRange<Int>]) {
        self.tensor = tensor
        self._ranges = ranges
    }
    
    public var shape: Shape {
        return Shape(_ranges.map { Dimension($0.count) } )
    }
}

extension TensorSlice {
    
    public subscript(indices: Int...) -> Element {
        get {
            return tensor._subscript(indices)
        }
    }
    
    public func volume() -> Int {
        return shape.volume()
    }
}

extension TensorSlice {
    
    public subscript(ranges: Range<Int>...) -> TensorSlice {
        return tensor._subscript(ranges: ranges.map { CountableRange($0) })
    }
    
    public subscript(ranges: ClosedRange<Int>...) -> TensorSlice {
        return tensor._subscript(ranges: ranges.map { CountableRange($0) })
    }
    
    public subscript(ranges: CountableRange<Int>...) -> TensorSlice {
        return tensor._subscript(ranges: ranges)
    }
    
    public subscript(ranges: CountableClosedRange<Int>...) -> TensorSlice {
        return tensor._subscript(ranges: ranges.map { CountableRange($0) })
    }
    
    public subscript(ranges: Range<Int>?...) -> TensorSlice {
        let validRanges = ranges.enumerated().map { i, range in
            range.map(CountableRange.init) ?? 0..<self.shape.dimensions[i].value
        }
        return tensor._subscript(ranges: validRanges)
    }
    
    public subscript(ranges: ClosedRange<Int>?...) -> TensorSlice {
        let validRanges = ranges.enumerated().map { i, range in
            range.map(CountableRange.init) ?? 0..<self.shape.dimensions[i].value
        }
        return tensor._subscript(ranges: validRanges)
    }
    
    public subscript(ranges: CountableRange<Int>?...) -> TensorSlice {
        let validRanges = ranges.enumerated().map { i, range in
            range ?? 0..<self.shape.dimensions[i].value
        }
        return tensor._subscript(ranges: validRanges)
    }
    
    public subscript(ranges: CountableClosedRange<Int>?...) -> TensorSlice {
        let validRanges = ranges.enumerated().map { i, range in
            range.map(CountableRange.init) ?? 0..<self.shape.dimensions[i].value
        }
        return tensor._subscript(ranges: validRanges)
    }

}

extension TensorSlice : Sequence {
    public func makeIterator() -> ElementGenerator {
        
        let validRanges = zip(_ranges, self.tensor.shape.dimensions).map { range, max in
            validRange(range, maxValue: max.value)
        }
        
        return ElementGenerator(tensor: tensor, ranges: validRanges)
    }
}

private func validRange(_ range: CountableRange<Int>, maxValue: Int) -> CountableRange<Int> {
    return max(0, range.lowerBound)..<min(maxValue, range.upperBound)
}


public struct ElementGenerator : IteratorProtocol {
    
    public typealias Element = Float
    
    fileprivate let tensor: Tensor
    
    fileprivate let ranges: [CountableRange<Int>]
    
    private var index: [Int]
    
    init(tensor: Tensor, ranges: [CountableRange<Int>]) {
        self.tensor = tensor
        self.ranges = ranges
        
        index = ranges.map { $0.lowerBound }
    }
    
    public mutating func next() -> Float? {
        for i in (-1..<index.count).reversed() {
            if index[i] < ranges[i].upperBound {
                break
            }
            if i == 0 {
                return nil
            }
            index[i] = ranges[i].lowerBound
            index[i-1] += 1
        }
        
        defer { index[index.count-1] += 1 }
        return tensor._subscript(index)
    }
}

extension TensorSlice: Equatable {}
public func ==(lhs: TensorSlice, rhs: TensorSlice) -> Bool {
    guard lhs.shape == rhs.shape else {
        return false
    }
    for (l, r) in zip(lhs, rhs) {
        guard l==r else {
            return false
        }
    }
    return true
}


extension Int {
    internal func ceilDiv(rhs: Int) -> Int {
        return (self + rhs - 1) / rhs
    }
}

module kernel-set-example-rmsnorm

go 1.21

require github.com/kernel-set/go v0.0.0

// Use the in-tree Go binding. When consuming kernel-set from elsewhere, drop
// this replace and `go get github.com/kernel-set/go`.
replace github.com/kernel-set/go => ../../bindings/go

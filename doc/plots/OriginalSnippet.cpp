for (int i=0; i<localNumberOfRows; i++)
		double rowDot = 0.0;
		int diag_tmp;
		diag_tmp=A.values(diag(i));
		z_new(i)=r(i)/diag_tmp;
		z_new(i) += z_old(i);
		for(int k = A.graph.row_map(i); k <= diag(i); k++)
			    rowDot += A.values(k) * z_old(A.graph.entries(k));
        z_new(i) -=rowDot/diag_tmp;
	}
}